import { cva, type VariantProps } from 'class-variance-authority'
import type * as React from 'react'

import { cn } from '@/lib/utils'

const iconBadgeVariants = cva(
  'inline-flex shrink-0 items-center justify-center rounded-md',
  {
    variants: {
      tone: {
        primary: 'bg-accent text-primary',
        chart1: 'bg-chart-1/12 text-chart-1',
        chart2: 'bg-chart-2/12 text-chart-2',
        chart3: 'bg-chart-3/12 text-chart-3',
        chart4: 'bg-chart-4/12 text-chart-4',
        chart5: 'bg-chart-5/12 text-chart-5',
        success: 'bg-success/12 text-success',
        warning: 'bg-warning/15 text-warning',
        muted: 'bg-muted text-muted-foreground',
      },
      size: {
        sm: 'size-6 [&_svg]:size-3.5',
        default: 'size-7 [&_svg]:size-4',
        lg: 'size-8 [&_svg]:size-[18px]',
      },
    },
    defaultVariants: {
      tone: 'primary',
      size: 'default',
    },
  }
)

export interface IconBadgeProps
  extends React.ComponentProps<'span'>,
    VariantProps<typeof iconBadgeVariants> {}

/** 卡片标题前的小色块图标（§2.4）。 */
export function IconBadge({ className, tone, size, ...props }: IconBadgeProps) {
  return (
    <span
      data-slot="icon-badge"
      className={cn(iconBadgeVariants({ tone, size }), className)}
      {...props}
    />
  )
}

export { iconBadgeVariants }
