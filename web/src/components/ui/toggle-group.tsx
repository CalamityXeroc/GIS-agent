import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group'
import type * as React from 'react'

import { cn } from '@/lib/utils'

/** 分段控件：图表卡右上角的「柱状 / 面积」切换用它（§2.5 第 3 条）。 */
export function ToggleGroup({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Root>) {
  return (
    <ToggleGroupPrimitive.Root
      className={cn(
        'inline-flex items-center gap-0.5 rounded-full border border-border bg-card p-0.5',
        className
      )}
      {...props}
    />
  )
}

export function ToggleGroupItem({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Item>) {
  return (
    <ToggleGroupPrimitive.Item
      className={cn(
        'inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors outline-none',
        'hover:text-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground',
        className
      )}
      {...props}
    />
  )
}
