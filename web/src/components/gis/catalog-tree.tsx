import {
  ChevronRight,
  Database,
  FileSpreadsheet,
  Folder,
  FolderOpen,
  Grid3x3,
  HardDrive,
  Layers,
  Plus,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { DatasetItem } from '@/lib/api'
import { cn } from '@/lib/utils'

import { crsBadge, formatNumber, kindLabel } from './format'
import { isNonSpatialTable } from './store'
import { defaultOpenKeys, type CatalogNode } from './tree'

export interface CatalogTreeProps {
  root: CatalogNode
  selectedPath: string | null
  addedPaths: Set<string>
  /** 每次 +1 把树折叠到只剩根节点（“全部折叠”按钮）。 */
  collapseSeq?: number
  onSelect: (item: DatasetItem) => void
  onAdd: (item: DatasetItem) => void
}

function NodeIcon({ node, open }: { node: CatalogNode; open: boolean }) {
  const className = 'size-3.5 shrink-0'
  switch (node.kind) {
    case 'root':
      return <FolderOpen className={cn(className, 'text-warning')} />
    case 'gdb':
      return <HardDrive className={cn(className, 'text-primary')} />
    case 'folder':
      return open ? (
        <FolderOpen className={cn(className, 'text-warning')} />
      ) : (
        <Folder className={cn(className, 'text-warning')} />
      )
    default:
      break
  }
  const kind = node.item?.kind
  if (kind === 'raster') return <Grid3x3 className={cn(className, 'text-chart-3')} />
  if (kind === 'table' || kind === 'project') {
    return <FileSpreadsheet className={cn(className, 'text-muted-foreground')} />
  }
  if (kind === 'other') return <Database className={cn(className, 'text-muted-foreground')} />
  return <Layers className={cn(className, 'text-success')} />
}

/** 降级徽标：`.gdb` 枚举不出成员、或名字含非法字符时给可读原因（悬停可见）。 */
function ReadErrorBadge({ error }: { error: string }) {
  return (
    <span
      className="rounded-[2px] bg-amber-500/15 px-1 text-[9px] leading-[13px] text-amber-600"
      title={error}
    >
      不可读
    </span>
  )
}

function Tail({ node }: { node: CatalogNode }) {
  const error = node.item?.error
  if (node.kind !== 'dataset') {
    return (
      <span className="flex shrink-0 items-center gap-1">
        {error ? <ReadErrorBadge error={error} /> : null}
        <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground/80">
          {formatNumber(node.count)}
        </span>
      </span>
    )
  }
  const badge = node.item ? crsBadge(node.item) : null
  const count = node.item?.feature_count
  // 非空间表（.gdb 里的 dbf / 统计表）没有几何：标「仅属性表」，双击只开表。
  const tableOnly = node.item ? isNonSpatialTable(node.item) : false
  return (
    <span className="flex shrink-0 items-center gap-1">
      {error ? (
        <ReadErrorBadge error={error} />
      ) : tableOnly ? (
        <span
          className="rounded-[2px] bg-muted px-1 text-[9px] leading-[13px] text-muted-foreground"
          title="非空间表：没有几何，只能看属性表"
        >
          仅属性表
        </span>
      ) : badge ? (
        <span
          className="rounded-[2px] bg-muted px-1 text-[9px] leading-[13px] tabular-nums text-muted-foreground"
          title={node.item?.crs_name || badge}
        >
          {badge}
        </span>
      ) : null}
      {count !== null && count !== undefined ? (
        <span className="text-[10px] tabular-nums text-muted-foreground/80">
          {formatNumber(count)}
        </span>
      ) : null}
    </span>
  )
}

/**
 * 目录树（spec §4 ①）：文件夹 → .gdb → 要素类，行高 24px。
 *
 * 单选 = 选中（右 dock 出元信息）；双击 = 加入图层栈；行内 + 同双击。
 */
export function CatalogTree({
  root,
  selectedPath,
  addedPaths,
  collapseSeq,
  onSelect,
  onAdd,
}: CatalogTreeProps) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(defaultOpenKeys(root)))
  const seeded = useRef(false)

  useEffect(() => {
    setOpen((previous) => {
      const next = new Set(previous)
      if (!seeded.current) {
        for (const key of defaultOpenKeys(root)) next.add(key)
      } else {
        next.add(root.key)
      }
      return next
    })
    seeded.current = true
  }, [root])

  useEffect(() => {
    if (!collapseSeq) return
    setOpen(new Set([root.key]))
  }, [collapseSeq, root.key])

  const toggle = (key: string) =>
    setOpen((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })

  const renderNode = (node: CatalogNode, level: number) => {
    const isFolder = node.kind !== 'dataset'
    const expanded = open.has(node.key)
    const selected = node.kind === 'dataset' && node.item?.path === selectedPath
    const added = node.kind === 'dataset' && node.item ? addedPaths.has(node.item.path) : false

    return (
      <div key={node.key}>
        <div
          role="treeitem"
          aria-level={level + 1}
          aria-expanded={isFolder ? expanded : undefined}
          aria-selected={selected}
          title={[node.item?.path ?? node.label, node.item?.error].filter(Boolean).join('\n')}
          onClick={() => {
            if (isFolder) toggle(node.key)
            else if (node.item) onSelect(node.item)
          }}
          onDoubleClick={(event) => {
            if (isFolder || !node.item) return
            event.preventDefault()
            onAdd(node.item)
          }}
          style={{ paddingLeft: 4 + level * 12 }}
          className={cn(
            'group flex h-[var(--gis-row-tree)] cursor-default items-center gap-1.5 rounded-[2px] pr-1 text-[12px] select-none',
            selected
              ? 'bg-accent font-medium text-accent-foreground'
              : 'text-foreground/85 hover:bg-accent/50'
          )}
        >
          {isFolder ? (
            <ChevronRight
              className={cn(
                'size-[11px] shrink-0 text-muted-foreground transition-transform',
                expanded && 'rotate-90'
              )}
            />
          ) : (
            <span className="w-[11px] shrink-0" aria-hidden="true" />
          )}
          <NodeIcon node={node} open={expanded} />
          <span className="min-w-0 flex-1 truncate">{node.label}</span>
          {added ? (
            <span
              className="size-1.5 shrink-0 rounded-full bg-primary"
              title="已在图层栈中"
              aria-hidden="true"
            />
          ) : null}
          <Tail node={node} />
          {node.kind === 'dataset' && node.item ? (
            <button
              type="button"
              title={
                isNonSpatialTable(node.item)
                  ? '打开属性表（非空间表，没有几何）'
                  : `添加到地图（${kindLabel(node.item.kind)}）`
              }
              aria-label={
                isNonSpatialTable(node.item)
                  ? `打开 ${node.label} 的属性表`
                  : `添加 ${node.label} 到地图`
              }
              onClick={(event) => {
                event.stopPropagation()
                onAdd(node.item as DatasetItem)
              }}
              className="grid size-[17px] shrink-0 place-items-center rounded-[2px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:bg-card hover:text-primary focus-visible:opacity-100"
            >
              <Plus className="size-[11px]" />
            </button>
          ) : null}
        </div>
        {isFolder && expanded
          ? node.children.map((child) => renderNode(child, level + 1))
          : null}
      </div>
    )
  }

  return (
    <div role="tree" aria-label="数据目录" className="flex flex-col gap-px">
      {renderNode(root, 0)}
    </div>
  )
}
