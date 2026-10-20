
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  DollarSign,
  GitBranch,
  History,
  LayoutDashboard,
  Play,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  XCircle,
  Zap,
} from 'lucide-react'

import {
  getBackendHealth,
  getExecutions,
  getExecutionTrace,
  getMetrics,
} from './api/dashboard'

import './App.css'

const navigation = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'executions', label: 'Executions', icon: Play },
  { id: 'traces', label: 'Trace Viewer', icon: GitBranch },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'history', label: 'History', icon: History },
]

const STATUS_LABELS = {
  ok: 'Completed',
  error: 'Failed',
  timeout: 'Timeout',
}

function formatCost(value) {
  if (value === null || value === undefined) return '—'

  const number = Number(value)

  if (!Number.isFinite(number)) return '—'

  return `$${number.toFixed(6)}`
}

function formatPercent(value) {
  if (value === null || value === undefined) return '—'

  const number = Number(value)

  if (!Number.isFinite(number)) return '—'

  return `${(number * 100).toFixed(1)}%`
}

function formatStepTime(value) {
  if (value === null || value === undefined) return '—'

  const number = Number(value)

  if (!Number.isFinite(number)) return '—'

  return `${number.toFixed(1)} ms`
}

function StatusBadge({ status }) {
  const success = status === 'ok'

  return (
    <span className={`status-badge ${success ? 'completed' : 'failed'}`}>
      {success ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
      {STATUS_LABELS[status] || status || 'Unknown'}
    </span>
  )
}

function MetricCard({ label, value, detail, icon: Icon, accent }) {
  return (
    <article className="metric-card">
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <span className={`metric-icon ${accent}`}>
          <Icon size={19} />
        </span>
      </div>

      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </article>
  )
}

function StatusBreakdown({ metrics }) {
  const total = metrics?.total_executions ?? 0

  const breakdown = [
    {
      label: 'Completed',
      count: metrics?.completed_executions ?? 0,
      color: '#50d5a4',
    },
    {
      label: 'Failed',
      count: metrics?.error_executions ?? 0,
      color: '#f39eab',
    },
    {
      label: 'Timed out',
      count: metrics?.timeout_executions ?? 0,
      color: '#f2b976',
    },
  ]

  return (
    <section className="panel activity-panel">
      <div className="panel-header">
        <div>
          <h2>Execution outcomes</h2>
          <p>Breakdown of saved C4 execution traces</p>
        </div>
        <Activity size={18} className="muted-icon" />
      </div>

      {total === 0 ? (
        <div className="empty-state">
          No execution outcomes recorded yet.
        </div>
      ) : (
        <div style={{ padding: '20px 24px 28px' }}>
          {breakdown.map((item) => {
            const percentage = (item.count / total) * 100

            return (
              <div
                key={item.label}
                style={{ marginBottom: 24 }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: 10,
                    fontSize: 12,
                  }}
                >
                  <span>{item.label}</span>
                  <strong>
                    {item.count} ({percentage.toFixed(1)}%)
                  </strong>
                </div>

                <div
                  style={{
                    height: 10,
                    background: '#283248',
                    borderRadius: 6,
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${percentage}%`,
                      height: '100%',
                      background: item.color,
                      borderRadius: 6,
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function SystemOverview({ connected, executionCount }) {
  return (
    <section className="panel system-panel">
      <div className="panel-header">
        <div>
          <h2>System overview</h2>
          <p>Backend integration status</p>
        </div>
        <Cpu size={18} className="muted-icon" />
      </div>

      <div className="system-items">
        <div className="system-item">
          <div className="system-item-icon">
            <Server size={19} />
          </div>

          <div>
            <strong>Dashboard API</strong>
            <span>FastAPI read-only service</span>
          </div>

          <span
            className={`system-state ${connected ? 'ready' : 'pending'}`}
          >
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        <div className="system-item">
          <div className="system-item-icon">
            <GitBranch size={19} />
          </div>

          <div>
            <strong>Execution traces</strong>
            <span>Validated JSON execution records</span>
          </div>

          <span className="system-state ready">
            {executionCount} saved
          </span>
        </div>

        <div className="system-item">
          <div className="system-item-icon">
            <BarChart3 size={19} />
          </div>

          <div>
            <strong>Quality measurements</strong>
            <span>Awaiting measurement integration</span>
          </div>

          <span className="system-state pending">
            Not available
          </span>
        </div>
      </div>

      <div className="system-footnote">
        <BrainCircuit size={18} />
        <span>
          Execution data is retrieved from the Neural-Core Python API.
        </span>
      </div>
    </section>
  )
}

function ExecutionTable({ rows, onSelect }) {
  return (
    <div className="table-scroll">
      <table className="execution-table">
        <thead>
          <tr>
            <th>Execution</th>
            <th>Framework</th>
            <th>Status</th>
            <th>Steps</th>
            <th>Cost</th>
            <th>Step time (sum)</th>
            <th />
          </tr>
        </thead>

        <tbody>
          {rows.map((run) => (
            <tr key={run.execution_id}>
              <td>
                <div className="execution-name">
                  {run.execution_id}
                </div>

                <div className="execution-id">
                  {run.model_ids?.join(', ') || 'No model recorded'}
                </div>
              </td>

              <td>{run.framework}</td>

              <td>
                <StatusBadge status={run.status} />
              </td>

              <td>{run.step_count}</td>

              <td>{formatCost(run.total_cost_usd)}</td>

              <td>{formatStepTime(run.total_step_wall_ms)}</td>

              <td>
                <button
                  type="button"
                  className="table-action"
                  onClick={() => onSelect(run)}
                  aria-label={`View trace for ${run.execution_id}`}
                >
                  <ChevronRight size={18} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {rows.length === 0 && (
        <div className="empty-state">
          No saved executions match the current filters.
        </div>
      )}
    </div>
  )
}

function TraceDetails({ execution, trace, loading, error }) {
  return (
    <section className="panel trace-panel">
      <div className="panel-header">
        <div>
          <h2>Execution trace</h2>
          <p>
            {execution
              ? execution.execution_id
              : 'Select a saved execution'}
          </p>
        </div>
        <GitBranch size={18} className="muted-icon" />
      </div>

      {loading && (
        <div className="empty-state">
          Loading execution trace...
        </div>
      )}

      {error && (
        <div className="empty-state" role="alert">
          {error}
        </div>
      )}

      {!loading && !error && !trace && (
        <div className="empty-state">
          Select an execution to view its recorded steps.
        </div>
      )}

      {!loading && !error && trace && (
        <>
          <div className="trace-timeline">
            {trace.steps.length === 0 && (
              <div className="empty-state">
                This execution has no recorded steps.
              </div>
            )}

            {trace.steps.map((step, index) => (
              <div
                className="trace-step"
                key={`${trace.execution_id}-${step.step_index}`}
              >
                <div className="trace-marker-wrap">
                  <span
                    className={`trace-marker ${
                      step.status !== 'ok' ? 'error' : ''
                    }`}
                  >
                    {step.status === 'ok' ? (
                      <CheckCircle2 size={13} />
                    ) : (
                      <XCircle size={13} />
                    )}
                  </span>

                  {index < trace.steps.length - 1 && (
                    <span className="trace-connector" />
                  )}
                </div>

                <div className="trace-step-content">
                  <strong>
                    {step.node_name ||
                      step.tool ||
                      `Execution step ${step.step_index}`}
                  </strong>

                  <span>Model: {step.model_id}</span>
                  <span>
                    Status: {STATUS_LABELS[step.status] || step.status}
                  </span>
                  <span>
                    Cost: {formatCost(step.cost_usd)}
                  </span>
                  <span>
                    Step duration: {formatStepTime(step.wall_ms)}
                  </span>
                  <span>
                    Tokens: {step.input_tokens} in / {step.output_tokens} out
                  </span>

                  {step.error_type && (
                    <span>Error type: {step.error_type}</span>
                  )}

                  {step.parent_steps?.length > 0 && (
                    <span>
                      Parent steps: {step.parent_steps.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="trace-footer">
            <span>Framework</span>
            <strong>{trace.framework}</strong>
          </div>

          <div className="trace-footer">
            <span>Final status</span>
            <StatusBadge status={trace.final_status} />
          </div>
        </>
      )}
    </section>
  )
}

function App() {
  const [activePage, setActivePage] = useState('overview')
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('All')
  const [refreshKey, setRefreshKey] = useState(0)

  const [dashboard, setDashboard] = useState({
    loading: true,
    error: null,
    health: null,
    metrics: null,
    executions: [],
  })

  const [selectedExecution, setSelectedExecution] = useState(null)

  const [traceState, setTraceState] = useState({
    loading: false,
    error: null,
    trace: null,
  })

  const traceRequestId = useRef(0)

  useEffect(() => {
    let cancelled = false

    Promise.all([
      getBackendHealth(),
      getExecutions(),
      getMetrics(),
    ])
      .then(([health, history, metrics]) => {
        if (cancelled) return

        if (health.status !== 'ok') {
          throw new Error('Backend health check failed')
        }

        setDashboard({
          loading: false,
          error: null,
          health,
          metrics,
          executions: Array.isArray(history.executions)
            ? history.executions
            : [],
        })
      })
      .catch((error) => {
        if (cancelled) return

        setDashboard({
          loading: false,
          error: error.message || 'Unable to reach the backend',
          health: null,
          metrics: null,
          executions: [],
        })
      })

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  const connected = dashboard.health?.status === 'ok'

  const filteredExecutions = useMemo(() => {
    return dashboard.executions.filter((run) => {
      const searchable = [
        run.execution_id,
        run.framework,
        ...(run.model_ids || []),
      ]
        .join(' ')
        .toLowerCase()

      const matchesSearch = searchable.includes(
        search.toLowerCase(),
      )

      const matchesStatus =
        statusFilter === 'All' ||
        run.status === statusFilter

      return matchesSearch && matchesStatus
    })
  }, [dashboard.executions, search, statusFilter])

  const activeLabel =
    navigation.find((item) => item.id === activePage)?.label ||
    'Overview'

  function refreshDashboard() {
    traceRequestId.current += 1
    setSelectedExecution(null)
    setTraceState({
      loading: false,
      error: null,
      trace: null,
    })

    setDashboard((previous) => ({
      ...previous,
      loading: true,
      error: null,
    }))

    setRefreshKey((previous) => previous + 1)
  }

  async function openTrace(execution) {
    const currentRequest = ++traceRequestId.current

    setSelectedExecution(execution)
    setActivePage('traces')

    setTraceState({
      loading: true,
      error: null,
      trace: null,
    })

    try {
      const result = await getExecutionTrace(
        execution.execution_id,
      )

      if (currentRequest !== traceRequestId.current) return

      setTraceState({
        loading: false,
        error: null,
        trace: result.trace,
      })
    } catch (error) {
      if (currentRequest !== traceRequestId.current) return

      setTraceState({
        loading: false,
        error: error.message || 'Could not load execution trace',
        trace: null,
      })
    }
  }

  const showTable = [
    'overview',
    'executions',
    'history',
  ].includes(activePage)

  const metrics = dashboard.metrics

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={23} />
          </div>

          <div>
            <div className="brand-name">
              Neural<span>Core</span>
            </div>
            <div className="brand-subtitle">
              AGENT ORCHESTRATION
            </div>
          </div>
        </div>

        <div className="sidebar-content">
          <div className="nav-label">WORKSPACE</div>

          <nav className="nav-menu" aria-label="Dashboard navigation">
            {navigation.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`nav-item ${
                  activePage === id ? 'active' : ''
                }`}
                onClick={() => setActivePage(id)}
              >
                <Icon size={18} />
                <span>{label}</span>

                {activePage === id && (
                  <span className="nav-active-dot" />
                )}
              </button>
            ))}
          </nav>

          <div className="nav-label second-label">
            RESEARCH SYSTEM
          </div>

          <div className="sidebar-info">
            <span
              className="info-indicator"
              style={{
                background: connected ? '#50d5a4' : '#f1b95d',
              }}
            />

            <div>
              <strong>
                {dashboard.loading
                  ? 'Connecting...'
                  : connected
                    ? 'Backend connected'
                    : 'Backend disconnected'}
              </strong>

              <span>FastAPI dashboard service</span>
            </div>
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <ShieldCheck size={17} />
            <div>
              <strong>Pillar 4 · Execution</strong>
              <span>Read-only execution monitoring</span>
            </div>
          </div>

          <div className="sidebar-version">
            NEURAL-CORE · UI v0.2
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={15} />
            <strong>{activeLabel}</strong>
          </div>

          <div className="topbar-right">
            <span className="preview-indicator">
              <span
                style={{
                  background: connected ? '#50d5a4' : '#eab66d',
                }}
              />

              {dashboard.loading
                ? 'Connecting'
                : connected
                  ? 'API Connected'
                  : 'Disconnected'}
            </span>

            <div className="user-avatar">NC</div>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <Zap size={14} />
                EXECUTION INTELLIGENCE
              </div>

              <h1>{activeLabel}</h1>

              <p>
                Monitor Neural-Core execution traces,
                statuses, and recorded costs.
              </p>
            </div>

            <div style={{ display: 'flex', gap: 10 }}>
              <button
                type="button"
                className="new-execution-button"
                onClick={refreshDashboard}
                disabled={dashboard.loading}
                style={{ opacity: dashboard.loading ? 0.6 : 1 }}
              >
                <RefreshCw size={16} />
                Refresh
              </button>

              <button
                type="button"
                className="new-execution-button"
                disabled
                title="Execution creation is not available in the read-only API"
              >
                <Play size={16} />
                New execution
              </button>
            </div>
          </div>

          {dashboard.error && (
            <div className="demo-notice" role="alert">
              <XCircle size={17} />
              <span>
                Unable to load execution data: {dashboard.error}
              </span>
            </div>
          )}

          {dashboard.loading && (
            <div className="demo-notice" role="status">
              <RefreshCw size={17} />
              <span>Loading execution data from FastAPI...</span>
            </div>
          )}

          {(activePage === 'overview' ||
            activePage === 'analytics') && (
            <>
              <div className="metrics-grid">
                <MetricCard
                  label="Total executions"
                  value={
                    metrics
                      ? String(metrics.total_executions)
                      : '—'
                  }
                  detail="Saved execution traces"
                  icon={Activity}
                  accent="purple"
                />

                <MetricCard
                  label="Execution success rate"
                  value={formatPercent(metrics?.success_rate)}
                  detail="Completed executions / total"
                  icon={CheckCircle2}
                  accent="green"
                />

                <MetricCard
                  label="Total execution cost"
                  value={formatCost(metrics?.total_cost_usd)}
                  detail="Sum of recorded step costs"
                  icon={DollarSign}
                  accent="blue"
                />

                <MetricCard
                  label="Average latency"
                  value={
                    metrics?.average_latency_ms == null
                      ? '—'
                      : formatStepTime(metrics.average_latency_ms)
                  }
                  detail="End-to-end latency not yet available"
                  icon={Clock3}
                  accent="orange"
                />
              </div>

              <div className="content-grid">
                <StatusBreakdown metrics={metrics} />

                <SystemOverview
                  connected={connected}
                  executionCount={dashboard.executions.length}
                />
              </div>
            </>
          )}

          {showTable && (
            <section className="panel runs-panel">
              <div className="panel-header runs-header">
                <div>
                  <h2>
                    {activePage === 'overview'
                      ? 'Saved executions'
                      : activePage === 'history'
                        ? 'Execution history'
                        : 'All executions'}
                  </h2>

                  <p>
                    Validated traces from the Python execution backend
                  </p>
                </div>

                {activePage === 'overview' && (
                  <button
                    type="button"
                    className="text-action"
                    onClick={() => setActivePage('executions')}
                  >
                    View all <ArrowRight size={16} />
                  </button>
                )}
              </div>

              <div className="table-tools">
                <div className="search-box">
                  <Search size={17} />

                  <input
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)
                    }
                    placeholder="Search executions..."
                    aria-label="Search executions"
                  />
                </div>

                <div className="filter-buttons">
                  {[
                    ['All', 'All'],
                    ['ok', 'Completed'],
                    ['error', 'Failed'],
                    ['timeout', 'Timeout'],
                  ].map(([value, label]) => (
                    <button
                      type="button"
                      key={value}
                      className={
                        statusFilter === value ? 'selected' : ''
                      }
                      onClick={() => setStatusFilter(value)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <ExecutionTable
                rows={filteredExecutions}
                onSelect={openTrace}
              />
            </section>
          )}

          {activePage === 'traces' && (
            <div className="trace-view-layout">
              <section className="panel trace-selector-panel">
                <div className="panel-header">
                  <div>
                    <h2>Select execution</h2>
                    <p>Saved execution traces</p>
                  </div>
                </div>

                {dashboard.executions.length === 0 ? (
                  <div className="empty-state">
                    No execution traces have been saved yet.
                  </div>
                ) : (
                  <div className="trace-run-list">
                    {dashboard.executions.map((run) => (
                      <button
                        key={run.execution_id}
                        type="button"
                        className={`trace-run-option ${
                          run.execution_id ===
                          selectedExecution?.execution_id
                            ? 'selected'
                            : ''
                        }`}
                        onClick={() => openTrace(run)}
                      >
                        <span>
                          <strong>{run.execution_id}</strong>
                          <small>
                            {run.framework} · {run.step_count} steps
                          </small>
                        </span>

                        <ChevronRight size={18} />
                      </button>
                    ))}
                  </div>
                )}
              </section>

              <TraceDetails
                execution={selectedExecution}
                trace={traceState.trace}
                loading={traceState.loading}
                error={traceState.error}
              />
            </div>
          )}

          {activePage === 'analytics' && (
            <section className="panel analytics-detail">
              <div className="panel-header">
                <div>
                  <h2>Measurement analytics</h2>
                  <p>Recorded execution information</p>
                </div>

                <BarChart3 size={18} className="muted-icon" />
              </div>

              <div className="system-items">
                <div className="system-item">
                  <div className="system-item-icon">
                    <DollarSign size={19} />
                  </div>

                  <div>
                    <strong>Mean execution cost</strong>
                    <span>Average recorded cost per execution</span>
                  </div>

                  <strong>
                    {formatCost(metrics?.mean_cost_usd)}
                  </strong>
                </div>

                <div className="system-item">
                  <div className="system-item-icon">
                    <Activity size={19} />
                  </div>

                  <div>
                    <strong>Quality score</strong>
                    <span>Not included in current trace schema</span>
                  </div>

                  <strong>—</strong>
                </div>

                <div className="system-item">
                  <div className="system-item-icon">
                    <Clock3 size={19} />
                  </div>

                  <div>
                    <strong>End-to-end latency</strong>
                    <span>
                      Not available from current trace metrics
                    </span>
                  </div>

                  <strong>—</strong>
                </div>
              </div>
            </section>
          )}

          <footer className="dashboard-footer">
            <span>
              Neural-Core · Execution monitoring
            </span>

            <span>
              React + FastAPI · Read-only dashboard
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
