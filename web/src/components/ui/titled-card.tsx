import type * as React from 'react'

import { IconBadge, type IconBadgeProps } from '@/components/ui/icon-badge'
import { cn } from '@/lib/utils'

export interface TitledCardProps extends Omit<React.ComponentProps<'section'>, 'title'> {
  title: React.ReactNode
  /** 右上角的「总计 / 更新时间」等补充信息。 */
  meta?: React.ReactNode
  icon?: React.ComponentType<{ className?: string }>
  iconTone?: IconBadgeProps['tone']
  actions?: React.ReactNode
  bodyClassName?: string
  children?: React.ReactNode
}

/** 卡片壳：图标 + 标题 + 右侧信息，内容区可滚动。 */
export function TitledCard({
  title,
  meta,
  icon: Icon,
  iconTone = 'primary',
  actions,
  bodyClassName,
  className,
  children,
  ...props
}: TitledCardProps) {
  return (
    <section
      data-slot="titled-card"
      className={cn('overflow-hidden rounded-lg border border-border bg-card', className)}
      {...props}
    >
      <header className="flex items-center gap-2.5 px-4 py-3">
        {Icon ? (
          <IconBadge tone={iconTone} size="sm">
            <Icon />
          </IconBadge>
        ) : null}
        <h2 className="text-sm font-medium text-card-foreground">{title}</h2>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          {meta}
          {actions}
        </div>
      </header>
      <div className={cn('border-t border-border px-4 py-4', bodyClassName)}>{children}</div>
    </section>
  )
}
