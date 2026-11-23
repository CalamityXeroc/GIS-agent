import type * as React from 'react'

import { cn } from '@/lib/utils'

/** 骨架块：只做占位，不做位移动画，避免表格行高抖动。 */
export function Skeleton({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="skeleton"
      className={cn('animate-pulse rounded-md bg-[var(--skeleton-base)]', className)}
      {...props}
    />
  )
}

/** 多行文本骨架，宽度按行递减，读起来像一段话。 */
export function SkeletonText({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton
          key={index}
          className="h-3"
          style={{ width: `${100 - index * 12}%` }}
        />
      ))}
    </div>
  )
}
