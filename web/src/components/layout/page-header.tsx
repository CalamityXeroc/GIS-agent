import type * as React from 'react'

import { cn } from '@/lib/utils'

export interface PageHeaderProps extends Omit<React.ComponentProps<'div'>, 'title'> {
  title: React.ReactNode
  description?: React.ReactNode
  /** 右侧操作区（segmented control / 主按钮）。 */
  actions?: React.ReactNode
}

/** 页头：20px semibold 标题 + 12px 说明 + 右侧操作区（§2.1 内容区）。 */
export function PageHeader({
  title,
  description,
  actions,
  className,
  ...props
}: PageHeaderProps) {
  return (
    <div
      data-slot="page-header"
      className={cn('mb-5 flex flex-wrap items-start justify-between gap-3', className)}
      {...props}
    >
      <div className="min-w-0">
        <h1 className="text-xl leading-7 font-semibold tracking-tight text-foreground">
          {title}
        </h1>
        {description ? (
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  )
}
