import { useCallback, useEffect, useRef, useState } from 'react'

import { runEventsUrl } from '@/lib/api'
import { EVENT_NAMES, type RawRunEvent } from '@/lib/run-events'

/**
 * 订阅一个 run 的 SSE 事件流。
 *
 * - 初次连接带 `since=0`：服务端从环形缓冲回放（刷新页面时间线不丢）；
 * - 断线由浏览器 EventSource 自动重连并带 `Last-Event-ID`，服务端只补发缺失段；
 * - 事件按 seq 去重排序，重连叠出来的重复帧不会重复显示。
 */
export function useRunStream(runId: string | undefined) {
  const [events, setEvents] = useState<RawRunEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [truncated, setTruncated] = useState(false)
  const [finished, setFinished] = useState(false)
  const seen = useRef<Set<number>>(new Set())

  const reset = useCallback(() => {
    seen.current = new Set()
    setEvents([])
    setTruncated(false)
    setFinished(false)
  }, [])

  useEffect(() => {
    if (!runId) return
    reset()
    const source = new EventSource(runEventsUrl(runId, 0))

    const append = (raw: RawRunEvent) => {
      if (!raw || typeof raw.seq !== 'number') return
      if (seen.current.has(raw.seq)) return
      seen.current.add(raw.seq)
      setEvents((previous) => [...previous, raw].sort((a, b) => a.seq - b.seq))
    }

    const listeners: Array<[string, (event: MessageEvent) => void]> = EVENT_NAMES.map(
      (name) => [
        name,
        (message: MessageEvent) => {
          try {
            append(JSON.parse(message.data) as RawRunEvent)
          } catch {
            /* 坏帧跳过，时间线不能因为一条解析失败整页崩 */
          }
          if (name === 'run_end') {
            // 服务端发完 run_end 就按契约关连接；不主动 close 的话
            // EventSource 会每 2s 重连一次，把历史整段重放一遍。
            setFinished(true)
            setConnected(false)
            source.close()
          }
        },
      ]
    )

    const onHello = (message: MessageEvent) => {
      try {
        const data = JSON.parse(message.data) as { truncated?: boolean }
        setTruncated(Boolean(data.truncated))
      } catch {
        /* hello 只用于提示，解析失败无所谓 */
      }
    }

    const onOpen = () => setConnected(true)
    const onError = () => setConnected(false)

    source.addEventListener('hello', onHello as EventListener)
    for (const [name, handler] of listeners) {
      source.addEventListener(name, handler as EventListener)
    }
    source.addEventListener('open', onOpen)
    source.addEventListener('error', onError)

    return () => {
      for (const [name, handler] of listeners) {
        source.removeEventListener(name, handler as EventListener)
      }
      source.removeEventListener('hello', onHello as EventListener)
      source.removeEventListener('open', onOpen)
      source.removeEventListener('error', onError)
      source.close()
    }
  }, [runId, reset])

  return { events, connected, truncated, finished }
}
