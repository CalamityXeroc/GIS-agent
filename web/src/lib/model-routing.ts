/**
 * 「改模型名 → 角色路由跟着走」的唯一一处前端实现。
 *
 * 引擎里 `EngineLLMConfig.models_for()` 的候选顺序是「显式指定 → routing_rules →
 * model → fallback_models」，**路由排在 model 前面**：只改 `model` 不同步路由，
 * 10 个角色仍会先试旧模型，新 run 看着换了模型其实没换（DAIR-265 实测）。
 * 后端 `config_api.apply_patch()` 有同一套联动；这里再算一遍是为了让保存前的
 * diff 预览和「N 项未保存」跟真正会落盘的内容一致。
 */

export interface RoutingSyncInput {
  /** 磁盘上的 routing_rules（角色 → 候选模型）。 */
  routingRules: Record<string, string[]> | undefined
  /** 要检查的角色（顺序与页面表格一致）。 */
  roles: readonly string[]
  /** 这次保存里的新模型名。 */
  model: string
  /** 本次 patch 里显式给过的角色：以用户为准，不联动。 */
  explicit?: Record<string, string[]>
}

/**
 * 返回需要被改写的角色 → `[model]`（已等于、为空、或用户显式指定过的不返回）。
 */
export function syncedRoutingForModel({
  routingRules,
  roles,
  model,
  explicit,
}: RoutingSyncInput): Record<string, string[]> {
  const synced: Record<string, string[]> = {}
  if (!model) return synced
  for (const role of roles) {
    if (explicit?.[role]) continue
    const original = routingRules?.[role] ?? []
    if (original.length === 0) continue
    if (original.length === 1 && original[0] === model) continue
    synced[role] = [model]
  }
  return synced
}
