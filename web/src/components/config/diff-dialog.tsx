import { KeyRound, TriangleAlert } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { ConfigDiffRow } from '@/lib/api'

function formatValue(value: unknown): string {
  if (value === undefined) return '（无）'
  if (Array.isArray(value)) return value.length ? value.join(' → ') : '（空）'
  if (value !== null && typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export interface ConfigDiffDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  rows: ConfigDiffRow[]
  /** 只改了 API Key（diff 里只显示脱敏前后）。 */
  apiKeyChanged: boolean
  saving: boolean
  onConfirm: () => void
}

/** 保存前的一次性确认：红删绿增，逐键列出 before/after。 */
export function ConfigDiffDialog({
  open,
  onOpenChange,
  rows,
  apiKeyChanged,
  saving,
  onConfirm,
}: ConfigDiffDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>确认写入配置</DialogTitle>
          <DialogDescription>
            将改动 <span className="font-medium text-foreground">{rows.length}</span> 处并写入
            config/llm_config.json；写入前会自动备份，可从「备份」页还原。
            改动只对下一个 run 生效。
          </DialogDescription>
        </DialogHeader>

        <div className="mt-4 max-h-[45vh] space-y-2 overflow-y-auto">
          {rows.map((row) => (
            <div key={row.path} className="rounded-md border border-border bg-card px-3 py-2">
              <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                {row.secret ? <KeyRound className="size-3" /> : null}
                <span className="font-mono">{row.path}</span>
              </div>
              <div className="mt-1 grid gap-1 text-xs sm:grid-cols-2">
                <div className="rounded-sm bg-destructive/10 px-2 py-1 text-destructive">
                  <span className="mr-1 select-none">−</span>
                  <span className="break-all">{formatValue(row.before)}</span>
                </div>
                <div className="rounded-sm bg-success/10 px-2 py-1 text-success">
                  <span className="mr-1 select-none">+</span>
                  <span className="break-all">{formatValue(row.after)}</span>
                </div>
              </div>
            </div>
          ))}
          {rows.length === 0 && apiKeyChanged ? (
            <p className="text-xs text-muted-foreground">只更新 API Key（脱敏显示，不落日志）。</p>
          ) : null}
        </div>

        {apiKeyChanged ? (
          <p className="mt-3 flex items-start gap-1.5 text-[11px] text-muted-foreground">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning" />
            API Key 只显示后 4 位；服务端日志也只记录“是否改动”，不记录值。
          </p>
        ) : null}

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)} disabled={saving}>
            取消
          </Button>
          <Button size="sm" onClick={onConfirm} disabled={saving}>
            {saving ? '写入中…' : '确认保存'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
