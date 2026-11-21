import type { DatasetItem } from '@/lib/api'

/**
 * 把 catalog 的扁平 `path` 还原成「文件夹 → .gdb → 要素类」的目录树。
 *
 * - 用所有数据集的**公共目录前缀**当根（真实工作区里是 `.../workspace/input`），
 *   这样树里不会出现 `E:\BST_lab\...` 这种机器路径。
 * - `.gdb` 目录按文件夹处理，但用数据库图标、默认展开（它是要素类的容器）。
 * - `count` 是后代**数据集**数，不是子节点数；目录行尾显示它。
 */

export type CatalogNodeKind = 'root' | 'folder' | 'gdb' | 'dataset'

export interface CatalogNode {
  /** 相对根的稳定 key（用 `/` 归一），React key 与展开态都靠它。 */
  key: string
  label: string
  kind: CatalogNodeKind
  /** 仅 dataset 有：catalog 里那条记录（path 为原始写法）。 */
  item?: DatasetItem
  children: CatalogNode[]
  /** 后代数据集数量。 */
  count: number
}

function splitPath(path: string): string[] {
  return path.replace(/\\/g, '/').split('/').filter(Boolean)
}

function commonDirPrefix(items: DatasetItem[]): string[] {
  const dirs = items.map((item) => splitPath(item.path).slice(0, -1))
  if (dirs.length === 0) return []
  let prefix = dirs[0]
  for (const dir of dirs) {
    let index = 0
    while (
      index < prefix.length &&
      index < dir.length &&
      prefix[index].toLowerCase() === dir[index].toLowerCase()
    ) {
      index += 1
    }
    prefix = prefix.slice(0, index)
    if (prefix.length === 0) break
  }
  return prefix
}

function kindOfSegment(segment: string): 'folder' | 'gdb' {
  return segment.toLowerCase().endsWith('.gdb') ? 'gdb' : 'folder'
}

function sortChildren(children: CatalogNode[]): CatalogNode[] {
  return [...children].sort((a, b) => {
    const aFolder = a.kind !== 'dataset'
    const bFolder = b.kind !== 'dataset'
    if (aFolder !== bFolder) return aFolder ? -1 : 1
    return a.label.localeCompare(b.label, 'zh-CN', { numeric: true })
  })
}

function recount(node: CatalogNode): number {
  if (node.kind === 'dataset') {
    node.count = 1
    return 1
  }
  let total = 0
  for (const child of node.children) total += recount(child)
  node.count = total
  node.children = sortChildren(node.children)
  return total
}

export function buildCatalogTree(items: DatasetItem[]): CatalogNode {
  const prefix = commonDirPrefix(items)
  const rootLabel = prefix[prefix.length - 1] ?? 'workspace'
  const root: CatalogNode = {
    key: '__root__',
    label: rootLabel,
    kind: 'root',
    children: [],
    count: 0,
  }
  const index = new Map<string, CatalogNode>([[root.key, root]])

  for (const item of items) {
    const segments = splitPath(item.path)
    const relative = segments.slice(prefix.length)
    // 兜底：路径不在公共前缀下（理论上不会），至少别丢数据。
    const trail = relative.length > 0 ? relative : [item.name]
    let parent = root
    for (let depth = 0; depth < trail.length; depth += 1) {
      const isLeaf = depth === trail.length - 1
      const segment = trail[depth]
      const key = `${parent.key}/${segment}`
      let node = index.get(key)
      if (!node) {
        node = isLeaf
          ? {
              key,
              label: item.name,
              kind: 'dataset',
              item,
              children: [],
              count: 1,
            }
          : {
              key,
              label: segment,
              kind: kindOfSegment(segment),
              children: [],
              count: 0,
            }
        index.set(key, node)
        parent.children.push(node)
      }
      if (isLeaf) break
      parent = node
    }
  }

  recount(root)
  return root
}

/** 默认展开：根 + 所有 .gdb（数据库内部按图层组织，折起来多一次点击）。 */
export function defaultOpenKeys(root: CatalogNode): string[] {
  const keys = [root.key]
  const walk = (node: CatalogNode) => {
    if (node.kind === 'gdb') keys.push(node.key)
    for (const child of node.children) walk(child)
  }
  walk(root)
  return keys
}

/** 按名称/路径过滤；保留命中项的所有祖先。返回匹配的数据集数。 */
export function filterTree(root: CatalogNode, query: string): { tree: CatalogNode; matches: number } {
  const text = query.trim().toLowerCase()
  if (!text) return { tree: root, matches: root.count }

  let matches = 0
  const walk = (node: CatalogNode): CatalogNode | null => {
    if (node.kind === 'dataset') {
      const item = node.item
      const haystack = `${node.label} ${item?.path ?? ''}`.toLowerCase()
      if (!haystack.includes(text)) return null
      matches += 1
      return node
    }
    const children = node.children
      .map((child) => walk(child))
      .filter((child): child is CatalogNode => child !== null)
    if (children.length === 0) return null
    return { ...node, children, count: children.reduce((sum, child) => sum + child.count, 0) }
  }

  const tree = walk(root)
  return { tree: tree ?? { ...root, children: [], count: 0 }, matches }
}

/** 过滤命中时祖先必须全部展开，否则用户什么都看不到。 */
export function ancestorKeys(root: CatalogNode, path: string): string[] {
  const target = path.replace(/\\/g, '/')
  const keys: string[] = []
  const walk = (node: CatalogNode, trail: string[]): boolean => {
    if (node.kind === 'dataset') {
      const value = (node.item?.path ?? '').replace(/\\/g, '/')
      if (value === target) {
        keys.push(...trail)
        return true
      }
      return false
    }
    const next = [...trail, node.key]
    for (const child of node.children) {
      if (walk(child, next)) return true
    }
    return false
  }
  walk(root, [])
  return keys
}
