
import textwrap

def create_svg():
    # Configuration
    width = 1000
    height = 1400
    
    # Colors (Professional/Paper style - Refined)
    colors = {
        'bg': '#ffffff',
        'box_fill': '#f8f9fa',
        'box_stroke': '#343a40',
        'box_stroke_highlight': '#0d6efd', # Bootstrap Primary Blue
        'box_stroke_success': '#198754',   # Bootstrap Success Green
        'box_stroke_warning': '#ffc107',   # Bootstrap Warning Yellow
        'text': '#212529',
        'text_light': '#6c757d',
        'arrow': '#495057',
        'accent_blue': '#e7f5ff',
        'accent_green': '#e6fcf5',
        'accent_purple': '#f3f0ff',
        'accent_orange': '#fff4e6',
        'feedback_path': '#dc3545' # Red for feedback loop
    }
    
    svg_content = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="font-family: \'PingFang SC\', \'Microsoft YaHei\', \'Helvetica Neue\', Helvetica, Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="{colors["bg"]}"/>',
        
        # Definitions
        '<defs>',
        f'<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{colors["arrow"]}"/>',
        '</marker>',
        f'<marker id="arrow-feedback" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{colors["feedback_path"]}"/>',
        '</marker>',
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feGaussianBlur in="SourceAlpha" stdDeviation="3"/>',
        '<feOffset dx="3" dy="3" result="offsetblur"/>',
        '<feComponentTransfer><feFuncA type="linear" slope="0.15"/></feComponentTransfer>',
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>',
        '</filter>',
        '</defs>',
        
        # Title
        f'<text x="{width/2}" y="50" text-anchor="middle" font-size="28" font-weight="bold" fill="{colors["text"]}">AI-Philo-Scientist: 自动化理论生成与演化工作流</text>'
    ]

    # Helper function to escape XML special characters
    def escape_xml(text):
        if not isinstance(text, str):
            return text
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")

    # Helper function to draw a box
    def draw_box(x, y, w, h, title, subtitle=None, details=None, fill=colors['box_fill'], stroke=colors['box_stroke'], stroke_width=1.5, dashed=False, corner_radius=8):
        dash_attr = 'stroke-dasharray="5,5"' if dashed else ''
        title = escape_xml(title)
        content = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{corner_radius}" ry="{corner_radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {dash_attr} filter="url(#shadow)"/>',
            f'<text x="{x + w/2}" y="{y + 30}" text-anchor="middle" font-size="16" font-weight="bold" fill="{colors["text"]}">{title}</text>'
        ]
        current_y = y + 55
        if subtitle:
            subtitle = escape_xml(subtitle)
            # Increase width for wrapping to accommodate Chinese characters better
            lines = textwrap.wrap(subtitle, width=int(w/8)) 
            for line in lines:
                content.append(f'<text x="{x + w/2}" y="{current_y}" text-anchor="middle" font-size="13" font-style="italic" fill="{colors["text"]}">{line}</text>')
                current_y += 18
        
        if details:
            current_y += 10
            for item in details:
                item = escape_xml(item)
                content.append(f'<text x="{x + 15}" y="{current_y}" text-anchor="start" font-size="12" fill="{colors["text_light"]}">• {item}</text>')
                current_y += 16
                
        return content

    # Helper function to draw an arrow
    def draw_arrow(x1, y1, x2, y2, label=None, color=colors['arrow'], marker="arrow", curve=False):
        if curve:
            # Simple quadratic bezier for curve
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Offset control point to create curve
            ctrl_x = mid_x + (50 if x1 == x2 else 0)
            ctrl_y = mid_y 
            path = f'<path d="M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#{marker})"/>'
        else:
            path = f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#{marker})"/>'
            
        content = [path]
        if label:
            label = escape_xml(label)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Adjust label background size for Chinese
            content.append(f'<rect x="{mid_x - len(label)*7}" y="{mid_y - 12}" width="{len(label)*14}" height="24" fill="{colors["bg"]}" opacity="0.9"/>')
            content.append(f'<text x="{mid_x}" y="{mid_y + 5}" text-anchor="middle" font-size="12" font-weight="bold" fill="{color}">{label}</text>')
        return content

    # === Layout ===
    cx = width / 2
    
    # 1. INPUTS
    y_inputs = 100
    # Reduced height since details are removed
    svg_content.extend(draw_box(cx - 360, y_inputs, 240, 80, "先验理论库", "人类现有的科学知识", fill=colors['accent_blue']))
    svg_content.extend(draw_box(cx + 120, y_inputs, 240, 80, "LLM 认知引擎", "大语言模型核心", fill=colors['accent_purple']))
    
    # 2. THEORY GENERATION (Dual Path)
    y_gen_start = 240
    h_gen = 280 # Reduced height
    
    # Container
    svg_content.append(f'<rect x="40" y="{y_gen_start}" width="{width-80}" height="{h_gen}" rx="15" ry="15" fill="none" stroke="{colors["box_stroke_highlight"]}" stroke-width="2" stroke-dasharray="8,8"/>')
    svg_content.append(f'<text x="70" y="{y_gen_start + 35}" font-size="20" font-weight="bold" fill="{colors["box_stroke_highlight"]}">第一阶段：理论生成 (Theory Generation)</text>')
    
    # Path A: Contradiction
    svg_content.extend(draw_box(cx - 380, y_gen_start + 70, 300, 80, "矛盾驱动生成", "Direct Synthesis Method", fill=colors['box_fill']))
    
    # Path B: Concept Space
    svg_content.extend(draw_box(cx + 80, y_gen_start + 70, 300, 80, "概念空间驱动", "Unified Method", fill=colors['box_fill']))
    
    # Synthesis Engine (Convergence)
    svg_content.extend(draw_box(cx - 160, y_gen_start + 190, 320, 70, "理论合成引擎", "形式化与一致性检查", fill=colors['accent_orange']))
    
    # Arrows Input -> Gen
    svg_content.extend(draw_arrow(cx - 240, y_inputs + 80, cx - 240, y_gen_start + 70)) # Prior -> Contradiction
    svg_content.extend(draw_arrow(cx - 240, y_inputs + 80, cx + 150, y_gen_start + 70, curve=True)) # Prior -> Concept (Data)
    svg_content.extend(draw_arrow(cx + 240, y_inputs + 80, cx + 240, y_gen_start + 70)) # LLM -> Concept
    svg_content.extend(draw_arrow(cx + 240, y_inputs + 80, cx - 150, y_gen_start + 70, curve=True)) # LLM -> Contradiction
    
    # Arrows Gen -> Synthesis
    svg_content.extend(draw_arrow(cx - 230, y_gen_start + 150, cx - 100, y_gen_start + 190))
    svg_content.extend(draw_arrow(cx + 230, y_gen_start + 150, cx + 100, y_gen_start + 190))
    
    # 3. MULTI-LEVEL EVALUATION
    y_eval_start = y_gen_start + h_gen + 50
    h_eval = 300 # Reduced height
    
    # Container
    svg_content.append(f'<rect x="40" y="{y_eval_start}" width="{width-80}" height="{h_eval}" rx="15" ry="15" fill="none" stroke="{colors["box_stroke_highlight"]}" stroke-width="2" stroke-dasharray="8,8"/>')
    svg_content.append(f'<text x="70" y="{y_eval_start + 35}" font-size="20" font-weight="bold" fill="{colors["box_stroke_highlight"]}">第二阶段：多层评估 (Multi-Level Evaluation)</text>')
    
    # Exp Validation
    svg_content.extend(draw_box(cx - 420, y_eval_start + 70, 380, 90, "实验验证", "模拟关键物理实验", fill=colors['box_fill']))
    
    # Role Evaluation
    svg_content.extend(draw_box(cx + 40, y_eval_start + 70, 380, 90, "角色扮演评估", "AI专家评审团", fill=colors['box_fill']))
    
    # Scoring
    svg_content.extend(draw_box(cx - 160, y_eval_start + 200, 320, 70, "综合评分系统", "加权聚合指标", fill=colors['accent_orange']))
    
    # Arrows Synthesis -> Eval
    svg_content.extend(draw_arrow(cx, y_gen_start + 260, cx, y_eval_start + 70, label="候选理论"))
    # Split arrow
    svg_content.extend(draw_arrow(cx, y_eval_start + 60, cx - 230, y_eval_start + 70))
    svg_content.extend(draw_arrow(cx, y_eval_start + 60, cx + 230, y_eval_start + 70))
    
    # Arrows Eval -> Score
    svg_content.extend(draw_arrow(cx - 230, y_eval_start + 160, cx - 60, y_eval_start + 200))
    svg_content.extend(draw_arrow(cx + 230, y_eval_start + 160, cx + 60, y_eval_start + 200))
    
    # 4. EVOLUTION & FEEDBACK
    y_evo_start = y_eval_start + h_eval + 50
    h_evo = 240 # Reduced height
    
    # Container
    svg_content.append(f'<rect x="40" y="{y_evo_start}" width="{width-80}" height="{h_evo}" rx="15" ry="15" fill="none" stroke="{colors["box_stroke_success"]}" stroke-width="2" stroke-dasharray="8,8"/>')
    svg_content.append(f'<text x="70" y="{y_evo_start + 35}" font-size="20" font-weight="bold" fill="{colors["box_stroke_success"]}">第三阶段：演化与反馈 (Evolution &amp; Feedback)</text>')
    
    # Selection
    svg_content.extend(draw_box(cx - 380, y_evo_start + 70, 220, 80, "自然选择", "优胜劣汰机制", fill=colors['box_fill']))
    
    # Refinement
    svg_content.extend(draw_box(cx - 60, y_evo_start + 70, 220, 80, "精炼与变异", "演化算子", fill=colors['box_fill']))
    
    # Next Gen
    svg_content.extend(draw_box(cx + 260, y_evo_start + 70, 160, 80, "下一代理论", "迭代 N+1", fill=colors['accent_green']))
    
    # Arrows Score -> Evo
    svg_content.extend(draw_arrow(cx, y_eval_start + 270, cx - 270, y_evo_start + 70, label="已评分理论"))
    
    # Arrows Evo Internal
    svg_content.extend(draw_arrow(cx - 160, y_evo_start + 110, cx - 60, y_evo_start + 110))
    svg_content.extend(draw_arrow(cx + 160, y_evo_start + 110, cx + 260, y_evo_start + 110))
    
    # === FEEDBACK LOOP ===
    # From Next Gen back to Synthesis
    
    path_d = f"M {cx + 340} {y_evo_start + 70} L {cx + 340} {y_evo_start + 30} L {width - 40} {y_evo_start + 30} L {width - 40} {y_gen_start + 230} L {cx + 160} {y_gen_start + 230}"
    svg_content.append(f'<path d="{path_d}" stroke="{colors["feedback_path"]}" stroke-width="3" fill="none" marker-end="url(#arrow-feedback)" stroke-dasharray="10,5"/>')
    
    # Label for Feedback
    svg_content.append(f'<rect x="{width - 55}" y="{(y_evo_start + y_gen_start)/2 - 80}" width="30" height="220" fill="{colors["bg"]}" rx="5"/>')
    svg_content.append(f'<text x="{width - 40}" y="{(y_evo_start + y_gen_start)/2}" transform="rotate(90, {width - 40}, {(y_evo_start + y_gen_start)/2})" text-anchor="middle" font-size="16" font-weight="bold" fill="{colors["feedback_path"]}">演化反馈循环 (Refinement Loop)</text>')
    
    # 5. FINAL OUTPUT
    y_final = y_evo_start + 180
    svg_content.extend(draw_box(cx - 220, y_final, 440, 70, "最终优胜理论 (Winner Theory)", "例如: PRQM (得分: 0.907)", fill=colors['accent_green'], stroke=colors['box_stroke_success'], stroke_width=3))
    
    # Arrow to Final
    svg_content.extend(draw_arrow(cx - 270, y_evo_start + 150, cx - 220, y_final + 35, curve=True))
    
    svg_content.append('</svg>')
    
    with open('project_workflow_detailed.svg', 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    create_svg()
