import type { DatasetItem } from '@/lib/api'

/**
 * 把 catalog 的扁平 `path` 还原成「文件夹 → .gdb → 要素类」的目录树。
 *
 * - 用所有数据集的**公共目录前缀**当根（真实工作区里是 `.../workspace/input`），
 *   这样树里不会出现 `E:\…\workspace\input` 这种整条机器路径。
 * - `.gdb` 目录按文件夹处理，但用数据库图标、默认展开（它是要素类的容器）。
 *   容器自身也可能有一条 `kind='gdb'` 的 catalog 记录（DAIR-369 给"打不开也必须
 *   可见"用的），它不能变成盖住成员的叶子节点。
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
      const isGdb = kindOfSegment(segment) === 'gdb'
      let node = index.get(key)
      if (!node) {
        if (isLeaf && !isGdb) {
          node = {
            key,
            label: item.name,
            kind: 'dataset',
            item,
            children: [],
            count: 1,
          }
        } else {
          // .gdb 永远是文件夹：catalog 里它的成员是 path=`....gdb/要素类` 的记录。
          node = {
            key,
            label: isLeaf ? item.name : segment,
            kind: isGdb ? 'gdb' : 'folder',
            children: [],
            count: 0,
          }
          // 容器记录（kind='gdb'）用来带「不可读」原因与 tooltip，保留到节点上。
          if (isGdb) node.item = item
        }
        index.set(key, node)
        parent.children.push(node)
      } else if (isGdb && !node.item) {
        // 成员先到、容器记录后到（或反过来）：补上容器信息，不丢提示。
        node.item = item
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
