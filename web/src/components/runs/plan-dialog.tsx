import { useEffect, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ApiError, decidePlan, type PlanPayload } from '@/lib/api'

interface PlanDialogProps {
  runId: string
  plan: PlanPayload | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onDecided: () => void
}

/** 计划卡：go / edit / abort，决策前引擎阻塞在闸门上（超时自动开始）。 */
export function PlanDialog({ runId, plan, open, onOpenChange, onDecided }: PlanDialogProps) {
  const [revising, setRevising] = useState(false)
  const [revision, setRevision] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) {
      setRevising(false)
      setRevision('')
    }
  }, [open])

  const submit = async (action: 'start' | 'revise' | 'cancel') => {
    if (action === 'revise' && !revision.trim()) {
      toast.error('先写一句要改什么')
      return
    }
    setSaving(true)
    try {
      await decidePlan(runId, action, revision.trim())
      toast.success(action === 'cancel' ? '已取消，run 不会开始执行' : '决策已提交')
      onOpenChange(false)
      onDecided()
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : '决策提交失败')
    } finally {
      setSaving(false)
    }
  }

  const rows = plan?.rows ?? []
  const deadline = plan?.deadline ? new Date(plan.deadline) : null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>执行前确认计划</DialogTitle>
          <DialogDescription>
            引擎会停在计划卡上等你的决定；
            {deadline ? ` ${deadline.toLocaleTimeString('zh-CN', { hour12: false })} 前没有决定就自动开始。` : ' 超时后自动开始。'}
          </DialogDescription>
        </DialogHeader>

        <div className="mt-4 overflow-hidden rounded-md border border-border">
          {rows.length > 0 ? (
            <dl className="divide-y divide-border">
              {rows.map((row) => (
                <div key={row.label} className="grid grid-cols-[5rem_minmax(0,1fr)] gap-3 px-3 py-2">
                  <dt className="text-xs text-muted-foreground">{row.label}</dt>
                  <dd className="text-xs leading-relaxed whitespace-pre-wrap break-words text-foreground">
                    {row.value}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <pre className="max-h-64 overflow-auto px-3 py-2 font-sans text-xs leading-relaxed whitespace-pre-wrap break-words text-foreground">
              {plan?.text ?? '（计划卡内容为空）'}
            </pre>
          )}
        </div>

        {revising ? (
          <div className="mt-4">
            <label className="text-xs text-muted-foreground" htmlFor="plan-revision">
              要改什么（一句话）
            </label>
            <textarea
              id="plan-revision"
              value={revision}
              onChange={(event) => setRevision(event.target.value)}
              rows={3}
              placeholder="例如：只做分区统计，不生成地图"
              className="mt-1.5 w-full resize-none rounded-md border border-border bg-background px-3 py-2 text-xs leading-relaxed text-foreground outline-none focus-visible:border-primary"
            />
          </div>
        ) : null}

        <DialogFooter>
          {revising ? (
            <>
              <Button variant="ghost" onClick={() => setRevising(false)} disabled={saving}>
                返回
              </Button>
              <Button onClick={() => submit('revise')} disabled={saving}>
                {saving ? '提交中…' : '按修改后的计划开始'}
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" onClick={() => submit('cancel')} disabled={saving}>
                取消运行
              </Button>
              <Button variant="outline" onClick={() => setRevising(true)} disabled={saving}>
                改一句
              </Button>
              <Button onClick={() => submit('start')} disabled={saving}>
                {saving ? '提交中…' : '开始执行'}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
