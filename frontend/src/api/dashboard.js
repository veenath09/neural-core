
/**
 * Neural-Core Dashboard API Client
 *
 * Connects React to the existing FastAPI backend.
 * All data comes from actual backend responses.
 * No demonstration records are generated here.
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/+$/, '')

async function apiRequest(endpoint) {
  const controller = new AbortController()

  const timeout = setTimeout(() => {
    controller.abort()
  }, 10000)

  try {
    const response = await fetch(
      `${API_BASE_URL}${endpoint}`,
      {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
        signal: controller.signal,
      },
    )

    if (!response.ok) {
      throw new Error(
        `Backend request failed: HTTP ${response.status}`,
      )
    }

    return await response.json()
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(
        'Backend request timed out',
        { cause: error },
      )
    }

    throw error
  } finally {
    clearTimeout(timeout)
  }
}

export async function getBackendHealth() {
  return apiRequest('/api/health')
}

export async function getExecutions() {
  return apiRequest('/api/executions')
}

export async function getMetrics() {
  return apiRequest('/api/metrics')
}

export async function getExecutionTrace(executionId) {
  if (
    typeof executionId !== 'string' ||
    !executionId.trim()
  ) {
    throw new Error('Valid execution ID is required')
  }

  return apiRequest(
    `/api/executions/${encodeURIComponent(executionId)}`,
  )
}

export { API_BASE_URL }
