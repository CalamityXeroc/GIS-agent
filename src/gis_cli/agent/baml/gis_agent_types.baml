// GIS Agent BAML contract draft for cross-model output normalization.

class GISPlanStep {
  id string
  tool string
  description string
  input map<string, string>
  depends_on string[]
}

class GISExecutionPlan {
  id string
  goal string
  steps GISPlanStep[]
  expected_outputs string[]
}
