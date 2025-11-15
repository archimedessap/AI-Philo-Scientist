# Short-Card Workflow Prompts

本文件汇总短卡工作流在向 LLM 发送请求以生成新理论时使用的全部提示词模板，按调用顺序列出。占位符以 `{{UPPER_SNAKE_CASE}}` 表示。

## 1. 矛盾表生成 (`ContradictionAnalyzer._build_messages`)

### System
```
You are a quantum foundations researcher. Analyse the supplied interpretation cards. Compare their commitments and produce concise contradiction items. Use the provided JSON schema. Prefer deep, substantive disagreements over terminology.{{ OPTIONAL_TASK_FOCUS }}
```
其中 `{{ OPTIONAL_TASK_FOCUS }}` 在存在 `task_hint` 时追加：` Task focus: {{TASK_HINT}}`

### User
```
Interpretation cards:
{{FORMATTED_CARD_BLOCKS}}

For each contradiction: use card ids or names in fields 'A' and 'B'; pick the `issue` enum that fits best. The `one_line` text must be under 25 ASCII characters and explain the tension plainly.
```
`{{FORMATTED_CARD_BLOCKS}}` 由若干卡片片段拼接，每个片段格式如下：
```
[{INDEX}] id={{CARD_ID}} | name={{CARD_NAME}}
one_line: {{ONE_LINE}}
math_relation: type={{MATH_TYPE}} | math_change={{MATH_CHANGE}} | eq: {{EQUATIONS_SUMMARY}}
claims: {{KEY_CLAIMS_SEMICOLON_JOIN}}
born_rule: {{BORN_RULE}} | measurement: {{MEASUREMENT_UPDATE}} | locality: {{LOCALITY_NOTE}}
predictions: {{PREDICTIONS_TEXT}}
tags: {{TAG_LIST_COMMA_JOIN}}
```

## 2. 结构化新理论草案 (`UnifiedTheoryGenerator._build_card_machine_messages`)

### System
```
You design candidate quantum interpretations. Use the contradictions to extend theory space while respecting the constraints. Output only JSON that matches the provided schema.
```

### User
```
Selected cards:
{{FORMATTED_CARD_BLOCKS}}

Contradictions:
{{CONTRADICTION_ROWS}}

Hard constraints:
- {{HARD_RULE_1}}
- {{HARD_RULE_2}}
...
Nice-to-have goals:
- {{NICE_TO_HAVE_1}}
- {{NICE_TO_HAVE_2}}
...
```
- `{{FORMATTED_CARD_BLOCKS}}`：与上一节相同的卡片格式，但无编号前缀。
- `{{CONTRADICTION_ROWS}}`：来自矛盾表的逐行描述，形如 `A vs B [issue]: summary`，若无矛盾则写 `No contradictions returned.`。
- 约束列表按硬约束与偏好目标依次罗列。

## 3. 人类可读提案写作 (`UnifiedTheoryGenerator._build_card_human_messages`)

### System
```
Write a six-section human-readable proposal for a new quantum interpretation that resolves the listed contradictions. Each section should be one short paragraph (4-6 sentences) following this order: (1) Core commitments, (2) Relation to SQM mathematics, (3) Measurement and Born rule, (4) Ontology, (5) Distinct empirical or operational consequences, (6) Attitude toward Bell/Kochen-Specker/PBR.
```

### User
```
Cards considered:
{{FORMATTED_CARD_BLOCKS}}

Key contradictions:
{{CONTRADICTION_ROWS}}

Constraints to respect:
- {{CONSTRAINT_1}}
- {{CONSTRAINT_2}}
...
```
`{{CONSTRAINT_i}}` 列表由硬约束与 nice-to-have 条目顺序拼接；若列表为空则提供单行 `- None`。

---
以上提示词涵盖短卡工作流中 LLM 参与的全部阶段：矛盾识别、结构化理论生成以及人类可读写作。
