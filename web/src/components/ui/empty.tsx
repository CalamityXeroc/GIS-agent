import { Inbox, type LucideIcon } from 'lucide-react'
import type * as React from 'react'

import { IconBadge } from '@/components/ui/icon-badge'
import { cn } from '@/lib/utils'

export interface EmptyProps extends Omit<React.ComponentProps<'div'>, 'title'> {
  title: React.ReactNode
  description?: React.ReactNode
  icon?: LucideIcon
  action?: React.ReactNode
}

/** 空态：虚线卡片，居中图标 + 一句话说明 + 可选动作。 */
export function Empty({
  title,
  description,
  icon: Icon = Inbox,
  action,
  className,
  ...props
}: EmptyProps) {
  return (
    <div
      data-slot="empty"
      className={cn(
        'flex min-h-[360px] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card/50 px-6 py-14 text-center',
        className
      )}
      {...props}
    >
      <IconBadge tone="muted" size="lg">
        <Icon />
      </IconBadge>
      <div className="text-sm font-medium text-foreground">{title}</div>
      {description ? (
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
          {description}
        </p>
      ) : null}
      {action ? <div className="mt-1">{action}</div> : null}
    </div>
  )
}
