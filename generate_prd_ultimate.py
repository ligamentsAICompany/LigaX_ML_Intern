"""Generate the updated PRD PDF for LigaX ML Intern.

Captures the ultimate product goal: a MultiCloud Technology autonomous-ML platform
targeting four Indian enterprise pillars — ITR, GST, FieldOps, Call Center.
"""

from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

OUTPUT = r"D:\_AI_\LigaX_ML_Intern\huggingface-ml-intern-finetuning\PRD_LigaX_MultiCloud_Ultimate_Goal.pdf"

# Brand palette
PRIMARY = HexColor("#1f6feb")
ACCENT = HexColor("#8957e5")
DARK = HexColor("#0d1117")
MUTED = HexColor("#57606a")
LIGHT_BG = HexColor("#f6f8fa")
BORDER = HexColor("#d0d7de")
GREEN = HexColor("#1a7f37")
RED = HexColor("#cf222e")
AMBER = HexColor("#9a6700")
TEAL = HexColor("#0a7c8a")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleBig",
            parent=styles["Title"],
            fontSize=28,
            leading=34,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontSize=13,
            leading=17,
            textColor=PRIMARY,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=PRIMARY,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontSize=13.5,
            leading=18,
            textColor=DARK,
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3",
            parent=styles["Heading3"],
            fontSize=11.5,
            leading=15,
            textColor=ACCENT,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            textColor=DARK,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyMuted",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletItem",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            leftIndent=14,
            bulletIndent=2,
            spaceAfter=2,
            textColor=DARK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.8,
            leading=11.5,
            textColor=DARK,
            backColor=LIGHT_BG,
            borderColor=BORDER,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Pillar",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=DARK,
            leftIndent=4,
        )
    )
    return styles


STYLES = build_styles()


def p(text, style="Body"):
    return Paragraph(text, STYLES[style])


def bullet(text):
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", STYLES["BulletItem"])


def code(text):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe.replace("\n", "<br/>"), STYLES["CodeBlock"])


def kv_table(rows, col_widths=(40 * mm, 130 * mm)):
    tbl = Table(rows, colWidths=list(col_widths))
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
            ]
        )
    )
    return tbl


def header_table(rows, col_widths):
    """First row is header (filled), rest are body."""
    tbl = Table(rows, colWidths=list(col_widths))
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), LIGHT_BG]),
            ]
        )
    )
    return tbl


def priority_chip(level):
    color = {"P0": RED, "P1": AMBER, "P2": GREEN}[level]
    return Paragraph(
        f'<font color="{color.hexval()}"><b>{level}</b></font>', STYLES["Body"]
    )


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        20 * mm,
        12 * mm,
        "LigaX ML Intern  |  PRD v2.0  |  Ultimate Goal: MultiCloud AI for Indian Enterprise",
    )
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.restoreState()


# ---------------- Sections ----------------


def cover(story):
    story.append(Spacer(1, 40 * mm))
    story.append(p("Product Requirements Document", "TitleBig"))
    story.append(
        p(
            "LigaX ML Intern &mdash; MultiCloud Autonomous ML Agent for Indian Enterprise",
            "Subtitle",
        )
    )
    story.append(Spacer(1, 6 * mm))

    meta = [
        ["Version", "2.0 (Ultimate Goal edition)"],
        ["Status", "Active &mdash; supersedes parity PRD"],
        ["Owner", "Tushar Das (tushardas@ligaments.ai)"],
        ["Product line", "LigaX / Ligaments AI"],
        ["Date", date.today().isoformat()],
        ["Repository", "huggingface-ml-intern-finetuning"],
        ["Deployment target", "MultiCloud (HF Spaces, GCP Cloud Run, AWS, Azure)"],
        ["Primary domains", "ITR &middot; GST &middot; FieldOps &middot; Call Center"],
    ]
    rows = [[p(k, "BodyMuted"), p(v, "Body")] for k, v in meta]
    story.append(kv_table(rows))

    story.append(Spacer(1, 12 * mm))
    story.append(
        p(
            "This document defines the product vision, scope, and roadmap for the autonomous ML "
            "intern application living at <font name='Courier'>D:\\_AI_\\LigaX_ML_Intern\\huggingface-ml-intern-finetuning\\</font>. "
            "It supersedes the earlier parity PRD and reframes the project around its ultimate "
            "business goal: a MultiCloud AI platform that operationalises ML for Indian "
            "enterprise workflows across ITR, GST, FieldOps, and Call Center pillars.",
            "BodyMuted",
        )
    )
    story.append(PageBreak())


def executive_summary(story):
    story.append(p("1. Executive Summary", "H1"))
    story.append(
        p(
            "<b>LigaX ML Intern</b> is an autonomous, agent-driven machine-learning platform that lets a "
            "non-expert describe a business problem in natural language and receive a finetuned model, "
            "an evaluation report, and a deployed inference endpoint &mdash; without writing a line of "
            "training code. The agent runs an iterative tool-using loop over the Hugging Face "
            "ecosystem (Hub, Datasets, Jobs, Inference Router) to research, dataset-curate, train, "
            "evaluate, and ship models.",
            "Body",
        )
    )
    story.append(
        p(
            "The <b>ultimate goal</b> is to position this platform as <b>MultiCloud Technology&rsquo;s</b> "
            "default AI delivery layer for Indian enterprise problems &mdash; specifically across four "
            "high-value pillars: <b>Income Tax Return (ITR)</b> automation, <b>GST</b> workflows, "
            "<b>Field Operations (FieldOps)</b>, and <b>Call Center</b> support automation. Every "
            "engineering, model, and dataset decision should be evaluated through this lens.",
            "Body",
        )
    )
    story.append(
        p(
            "The Bitext-telco finetuning work currently in the repository is one concrete instance "
            "of the Call Center pillar &mdash; not the end goal. The end goal is a multi-tenant, "
            "multi-cloud product that ships purpose-built models for each pillar and lets customers "
            "operate them through a single web app.",
            "Body",
        )
    )


def vision(story):
    story.append(p("2. Product Vision", "H1"))
    story.append(
        p(
            "Build the <b>fastest path from an Indian enterprise problem statement to a production-ready, "
            "domain-specific model</b>. Where generic cloud-AI platforms expose primitives, LigaX ML "
            "Intern ships a pre-shaped vertical workflow per pillar &mdash; with curated base models, "
            "domain datasets, evaluation harnesses, and deployment recipes already wired in.",
            "Body",
        )
    )
    story.append(p("2.1 Positioning statement", "H2"))
    story.append(
        p(
            "<i>For Indian enterprises operating in tax, compliance, field-services, and customer-"
            "support domains, LigaX ML Intern is a MultiCloud autonomous ML platform that turns a "
            "natural-language brief into a finetuned, deployable model in hours rather than weeks &mdash; "
            "without requiring an in-house ML team.</i>",
            "Body",
        )
    )
    story.append(p("2.2 Why MultiCloud", "H2"))
    for line in [
        "Customers are heterogeneous: BFSI on AWS, government adjacencies on GCP, enterprise IT on Azure. Model artifacts must remain portable.",
        "Hugging Face Hub is the canonical model and dataset store; downstream serving lands on whichever cloud the customer mandates.",
        "Data residency: ITR and GST workflows touch PII and financial data. Some customers require in-country serving on a specific cloud region.",
        "Cost arbitrage: training on Hugging Face Jobs (T4/A10/A100 on demand), serving on the cheapest endpoint per tenant.",
    ]:
        story.append(bullet(line))


def domains(story):
    story.append(p("3. Target Domains &mdash; The Four Pillars", "H1"))
    story.append(
        p(
            "Every dataset pick, base-model recommendation, demo prompt, and evaluation harness "
            "should default to one of these four pillars. Generic English chatbot demos are out of scope.",
            "Body",
        )
    )

    rows = [
        ["Pillar", "Representative use cases", "Data sources", "Candidate base models"]
    ]
    rows.append(
        [
            p("<b>ITR</b><br/>Income Tax Return", "Pillar"),
            p(
                "ITR-1/2/4 prep assistance; Form 26AS / AIS reconciliation; deduction Q&amp;A; notice-response drafting; CA copilot.",
                "Pillar",
            ),
            p(
                "CBDT FAQs, ITR instruction PDFs, anonymised AIS samples, public CA forum threads, in-house labelled tickets.",
                "Pillar",
            ),
            p(
                "Gemma-2-9B, Llama-3.1-8B, IndicGemma, Sarvam-1; smaller distilled variants for on-device CA tooling.",
                "Pillar",
            ),
        ]
    )
    rows.append(
        [
            p("<b>GST</b><br/>Goods &amp; Services Tax", "Pillar"),
            p(
                "GSTR-1/3B reconciliation; HSN/SAC code lookup; ITC eligibility advisor; e-invoice validation; notice triage.",
                "Pillar",
            ),
            p(
                "GSTN circulars, CBIC FAQs, HSN master, public rulings, customer e-invoice corpora (anonymised).",
                "Pillar",
            ),
            p(
                "Gemma-2-2B/9B, Llama-3.1-8B; structured-output finetunes for JSON GSTR shapes.",
                "Pillar",
            ),
        ]
    )
    rows.append(
        [
            p("<b>FieldOps</b><br/>Field Operations", "Pillar"),
            p(
                "Field-technician dispatch optimisation; service-ticket summarisation; spare-part identification (multimodal); inspection-report generation; safety-checklist Q&amp;A.",
                "Pillar",
            ),
            p(
                "Dispatch logs, photo-attached service tickets, OEM manuals, multilingual technician transcripts.",
                "Pillar",
            ),
            p(
                "Gemma-2-2B (edge), Llama-3.2-3B, Phi-3.5-mini; VLMs for photo-based inspection (LLaVA, Qwen2-VL).",
                "Pillar",
            ),
        ]
    )
    rows.append(
        [
            p("<b>Call Center</b><br/>BFSI &amp; Telco", "Pillar"),
            p(
                "Agent-assist copilot; post-call summarisation; QA scoring; intent and sentiment routing; multilingual IVR fallback; complaint triage.",
                "Pillar",
            ),
            p(
                "Bitext-telco, anonymised BFSI call transcripts, complaint logs, multilingual CSAT corpora.",
                "Pillar",
            ),
            p(
                "Gemma-2-2B/9B, Llama-3.1-8B, IndicBERT for routing; <font name='Courier'>train_gemma_telco.py</font> in repo is the reference recipe.",
                "Pillar",
            ),
        ]
    )
    story.append(header_table(rows, col_widths=[28 * mm, 55 * mm, 45 * mm, 42 * mm]))


def personas(story):
    story.append(PageBreak())
    story.append(p("4. User Personas", "H1"))
    personas_data = [
        (
            "Chartered Accountant (CA) / Tax Professional",
            "Goal: scale ITR &amp; GST advisory without hiring more juniors.",
            "Wants finetuned models that know Indian tax law and produce structured drafts (notices, replies, computation sheets).",
            "Pillars: ITR, GST",
        ),
        (
            "Field-Service Manager (Utilities, Telco, BFSI on-ground)",
            "Goal: cut technician dispatch time and standardise inspection reports.",
            "Wants a multimodal model that reads field photos and produces a structured inspection log in English/Hindi/regional language.",
            "Pillar: FieldOps",
        ),
        (
            "Call-Center Operations Head (BFSI / Telco)",
            "Goal: improve AHT, FCR, and CSAT while reducing supervisor QA load.",
            "Wants agent-assist suggestions, automated QA scoring, and post-call summaries in the customer&rsquo;s language.",
            "Pillar: Call Center",
        ),
        (
            "Enterprise IT &amp; Compliance Lead",
            "Goal: deploy domain AI without leaking data outside the customer&rsquo;s cloud.",
            "Wants the same product to deploy on AWS / Azure / GCP with model weights pinned to a specific region.",
            "Pillars: All",
        ),
        (
            "ML / Data Engineer (LigaX internal &amp; partner ISVs)",
            "Goal: extend the platform with new datasets, evaluators, and pillars.",
            "Wants a clean agent SDK, tool plug-in interface, and reproducible HF Jobs scripts.",
            "Pillars: All",
        ),
    ]
    for title, goal, want, pillars in personas_data:
        story.append(p(title, "H3"))
        story.append(bullet(goal))
        story.append(bullet(want))
        story.append(bullet(f"<font color='{TEAL.hexval()}'><b>{pillars}</b></font>"))


def current_state(story):
    story.append(PageBreak())
    story.append(p("5. Current State (as of " + date.today().isoformat() + ")", "H1"))
    story.append(p("5.1 What ships today", "H2"))
    for line in [
        "<b>Autonomous agent loop</b> in <font name='Courier'>agent/</font> with up to 300 iterations, doom-loop detection, context auto-compaction at 170k tokens, and HF session upload.",
        "<b>Toolset</b>: HF docs &amp; research, repo/dataset/jobs/papers, GitHub code search, sandboxed code execution, planning, and MCP server tools.",
        "<b>FastAPI backend</b> at <font name='Courier'>backend/main.py</font> on port 7860 with SSE streaming, session management, tool-approval flow, and dev-mode auth bypass.",
        "<b>Frontend</b>: React + Vite app in <font name='Courier'>frontend/</font>; planned simplification to a single static HTML in <font name='Courier'>backend/static/</font> per IMPLEMENTATION.md.",
        "<b>Reference finetuning recipe</b>: <font name='Courier'>train_gemma_telco.py</font> against the Bitext telco dataset (Call Center pillar).",
        "<b>Container</b>: HF Spaces compatible Dockerfile exposing port 7860; Cloud Run deployable.",
    ]:
        story.append(bullet(line))

    story.append(p("5.2 Known gaps vs. the ultimate goal", "H2"))
    for line in [
        "Pillar specialisation is implicit, not explicit &mdash; no UI pillar selector, no per-pillar dataset/model catalogue, no per-pillar eval harness.",
        "No Indian-language defaults: prompts, datasets, and base models still skew English-only.",
        "No PII / data-residency guardrails for ITR and GST workloads.",
        "MultiCloud deployment story is documented for Cloud Run only; AWS and Azure recipes are missing.",
        "Evaluation is ad-hoc; no domain-specific benchmarks per pillar.",
        "Multi-tenant quotas and isolation (<font name='Courier'>backend/user_quotas.py</font>) exist but are unproven at scale.",
    ]:
        story.append(bullet(line))


def goals_nongoals(story):
    story.append(PageBreak())
    story.append(p("6. Goals and Non-Goals", "H1"))
    story.append(p("6.1 Goals", "H2"))
    for line in [
        "<b>Pillar-first UX</b>: the user picks ITR / GST / FieldOps / Call Center as the first action; everything downstream is pre-shaped for that pillar.",
        "<b>Curated catalogues</b> per pillar: base models, datasets, eval suites, demo prompts, and deployment templates.",
        "<b>Indian-language readiness</b>: Hindi + at least 4 regional languages supported in inference paths for FieldOps and Call Center.",
        "<b>MultiCloud deployment</b>: published reference deployments on HF Spaces, GCP Cloud Run, AWS (ECS / SageMaker), and Azure (Container Apps / AML).",
        "<b>Compliance posture</b>: PII redaction in agent traces, region-pinned training, audit log of every tool call for ITR/GST workloads.",
        "<b>Reproducibility</b>: every finetune produces a model card + dataset snapshot + HF Jobs run record.",
    ]:
        story.append(bullet(line))

    story.append(p("6.2 Non-Goals (explicit)", "H2"))
    for line in [
        "We are <b>not</b> building a general-purpose ChatGPT competitor.",
        "We are <b>not</b> training foundation models from scratch &mdash; we finetune open-weight bases.",
        "We are <b>not</b> targeting non-Indian-context use cases in v2.x; English-only global chatbot demos are out of scope.",
        "We are <b>not</b> building our own GPU fleet; HF Jobs (and customer-cloud GPUs where required) own training compute.",
        "We are <b>not</b> shipping a mobile-native app in v2.x &mdash; web only.",
    ]:
        story.append(bullet(line))


def functional_requirements(story):
    story.append(PageBreak())
    story.append(p("7. Functional Requirements", "H1"))

    story.append(p("7.1 Cross-cutting requirements", "H2"))
    cross_cutting = [
        (
            "CC-1",
            "P0",
            "Pillar selector on the landing screen; agent system prompt is pillar-conditioned.",
        ),
        (
            "CC-2",
            "P0",
            "Per-pillar dataset catalogue surfaced as searchable picker (backed by HF datasets API + curated list).",
        ),
        (
            "CC-3",
            "P0",
            "Per-pillar base-model catalogue with recommended hyperparameters and LoRA defaults.",
        ),
        (
            "CC-4",
            "P0",
            "Training jobs submitted to HF Jobs with pillar tag; results pushed to a pillar-namespaced HF org.",
        ),
        (
            "CC-5",
            "P1",
            "PII redaction pass over user prompts and tool outputs for ITR / GST pillars.",
        ),
        (
            "CC-6",
            "P1",
            "Per-pillar evaluation suite runs automatically post-train and writes a model-card eval table.",
        ),
        (
            "CC-7",
            "P1",
            "Multi-tenant quotas: per-tenant token budget, concurrent-job cap, and storage cap.",
        ),
        (
            "CC-8",
            "P2",
            "Pillar-tagged session logs queryable from <font name='Courier'>session_logs/</font>.",
        ),
    ]
    rows = [["ID", "Pri", "Requirement"]]
    for rid, pr, desc in cross_cutting:
        rows.append([p(f"<b>{rid}</b>", "Body"), priority_chip(pr), p(desc, "Body")])
    story.append(header_table(rows, col_widths=[18 * mm, 14 * mm, 138 * mm]))

    pillar_reqs = [
        (
            "7.2 ITR pillar",
            [
                (
                    "ITR-1",
                    "P0",
                    "ITR-1/2/4 form-field Q&amp;A finetune on CBDT instruction corpus + curated CA-forum threads.",
                ),
                (
                    "ITR-2",
                    "P0",
                    "Structured-output mode: produce JSON matching the ITR XML schema for selected sections.",
                ),
                (
                    "ITR-3",
                    "P1",
                    "Notice-response drafting from a notice PDF + taxpayer context; cites the relevant section/rule.",
                ),
                (
                    "ITR-4",
                    "P1",
                    "Form 26AS / AIS reconciliation against bank-statement uploads.",
                ),
                ("ITR-5", "P2", "Multi-year-comparison report generator."),
            ],
        ),
        (
            "7.3 GST pillar",
            [
                (
                    "GST-1",
                    "P0",
                    "HSN/SAC code lookup with confidence and source citation.",
                ),
                (
                    "GST-2",
                    "P0",
                    "GSTR-1 vs. GSTR-3B reconciliation finetune; flags mismatches with reason codes.",
                ),
                (
                    "GST-3",
                    "P1",
                    "ITC eligibility advisor: input vendor invoices, return eligibility verdict and rationale.",
                ),
                (
                    "GST-4",
                    "P1",
                    "e-invoice JSON validation and auto-correction recommendations.",
                ),
                (
                    "GST-5",
                    "P2",
                    "Notice triage: classify GST notices and suggest first-response template.",
                ),
            ],
        ),
        (
            "7.4 FieldOps pillar",
            [
                (
                    "FO-1",
                    "P0",
                    "Service-ticket summarisation finetune on multilingual ticket logs.",
                ),
                (
                    "FO-2",
                    "P0",
                    "Photo-to-report: VLM finetune that ingests field photos + free-text notes and emits a structured inspection report.",
                ),
                (
                    "FO-3",
                    "P1",
                    "Dispatch-priority scorer from ticket text + SLA metadata.",
                ),
                (
                    "FO-4",
                    "P1",
                    "Spare-part identification from image (closed-set classifier + open-set fallback).",
                ),
                ("FO-5", "P2", "Safety-checklist Q&amp;A copilot for technicians."),
            ],
        ),
        (
            "7.5 Call Center pillar",
            [
                (
                    "CC1-1",
                    "P0",
                    "Post-call summarisation finetune (Bitext-telco baseline already in repo).",
                ),
                (
                    "CC1-2",
                    "P0",
                    "Intent + sentiment router for incoming chats and call transcripts.",
                ),
                (
                    "CC1-3",
                    "P1",
                    "Agent-assist suggestion engine: surfaces next-best-action and KB snippets in real time.",
                ),
                (
                    "CC1-4",
                    "P1",
                    "Automated QA scoring against a configurable rubric; writes to a QA dashboard.",
                ),
                (
                    "CC1-5",
                    "P2",
                    "Multilingual IVR fallback model (Hindi + 4 regional).",
                ),
            ],
        ),
    ]
    for header, items in pillar_reqs:
        story.append(p(header, "H2"))
        rows = [["ID", "Pri", "Requirement"]]
        for rid, pr, desc in items:
            rows.append(
                [p(f"<b>{rid}</b>", "Body"), priority_chip(pr), p(desc, "Body")]
            )
        story.append(header_table(rows, col_widths=[18 * mm, 14 * mm, 138 * mm]))


def architecture(story):
    story.append(PageBreak())
    story.append(p("8. Technical Architecture", "H1"))

    story.append(p("8.1 High-level component map", "H2"))
    story.append(
        code(
            "┌────────────────────────────────────────────────────────────────┐\n"
            "│  Web UI (single-page HTML in backend/static/, served by FastAPI)│\n"
            "│  Screens: Pillar select → Dataset/Model → Train → Chat          │\n"
            "└──────────────┬──────────────────────────────────────────────────┘\n"
            "               │ HTTPS + SSE\n"
            "┌──────────────▼──────────────────────────────────────────────────┐\n"
            "│  FastAPI backend (backend/main.py, port 7860)                   │\n"
            "│  • /api/session, /api/chat/{id}, /api/platform/*                │\n"
            "│  • Session manager, quotas, OAuth (HF) + dev bypass             │\n"
            "└──────────────┬──────────────────────────────────────────────────┘\n"
            "               │ in-process\n"
            "┌──────────────▼──────────────────────────────────────────────────┐\n"
            "│  Agent runtime (agent/, installed editable as hf-agent)         │\n"
            "│  • agent.core.agent_loop (submission_loop + handlers)           │\n"
            "│  • ContextManager, ToolRouter, DoomLoopDetector                 │\n"
            "│  • Pillar-conditioned system prompt + per-pillar tool subset    │\n"
            "└──────────────┬──────────────────────────────────────────────────┘\n"
            "               │\n"
            "        ┌──────┴──────┬──────────────┬────────────────┐\n"
            "        ▼             ▼              ▼                ▼\n"
            "   HF Hub / Datasets  HF Jobs    HF Router       Customer Cloud\n"
            "   (models + data)    (training) (inference)     (region-pinned\n"
            "                                                  serving)\n"
        )
    )

    story.append(p("8.2 Pillar conditioning", "H2"))
    story.append(
        p(
            "The agent receives a <font name='Courier'>pillar</font> field on session creation. The system prompt, "
            "tool allowlist, dataset picker, and base-model recommender are all conditioned on this field. "
            "Pillar metadata is persisted with the session and tagged onto every HF Jobs run, model card, "
            "and dataset push.",
            "Body",
        )
    )

    story.append(p("8.3 MultiCloud deployment shapes", "H2"))
    rows = [["Cloud", "Compute target", "Status", "Notes"]]
    rows += [
        [
            "HF Spaces",
            "ZeroGPU / CPU upgrade",
            "Live (ref impl.)",
            "Default demo deploy; port 7860; Dockerfile in repo.",
        ],
        [
            "GCP Cloud Run",
            "1× vCPU 2GiB+, Cloud Run Jobs for offline batches",
            "Live (ref impl.)",
            "60-min request cap; suitable for inference proxy + agent.",
        ],
        [
            "AWS ECS Fargate + SageMaker Endpoints",
            "Fargate for app, SageMaker JumpStart for serving HF models",
            "Planned v2.2",
            "Use HF DLC; private VPC for ITR/GST.",
        ],
        [
            "Azure Container Apps + AML Endpoints",
            "Container Apps for app, AML Managed Online Endpoints for serving",
            "Planned v2.3",
            "BYO key vault for HF token + customer secrets.",
        ],
        [
            "On-prem / customer VPC",
            "Docker Compose or K8s",
            "Roadmap",
            "Required for some BFSI / government customers.",
        ],
    ]
    cells = []
    for i, r in enumerate(rows):
        cells.append([p(x, "Body") if i > 0 else x for x in r])
    story.append(header_table(cells, col_widths=[35 * mm, 50 * mm, 30 * mm, 55 * mm]))

    story.append(p("8.4 Data flow &mdash; train and deploy", "H2"))
    for line in [
        "User picks pillar → UI shows pillar-curated dataset picker + base-model picker.",
        "User uploads or selects dataset → backend pushes to private HF dataset repo (PII-redacted for ITR/GST).",
        "User submits training prompt → agent constructs TRL/SFT script, submits to HF Jobs with pillar tag.",
        "Job logs streamed to UI via SSE; model + tokenizer pushed to a pillar-namespaced HF org on completion.",
        "Eval harness runs against the pillar benchmark; eval table written to the model card.",
        "Inference: HF Router by default; customer-cloud endpoint deployed via per-cloud recipe on request.",
    ]:
        story.append(bullet(line))


def data_strategy(story):
    story.append(PageBreak())
    story.append(p("9. Data Strategy", "H1"))
    story.append(
        p(
            "Each pillar owns a curated dataset bundle on the LigaX HF org. New training runs default "
            "to the bundle; users can extend with their own uploads.",
            "Body",
        )
    )

    rows = [["Pillar", "Default bundle (HF dataset slug)", "Modality", "Status"]]
    rows += [
        [
            "ITR",
            "ligaments/itr-faq-cbdt-v1 (+ ligaments/itr-notice-replies-v1)",
            "Text",
            "v1 in curation",
        ],
        [
            "GST",
            "ligaments/gst-circulars-v1 (+ ligaments/gst-hsn-master-v1)",
            "Text + structured",
            "v1 in curation",
        ],
        [
            "FieldOps",
            "ligaments/fieldops-tickets-multiling-v1 (+ ligaments/fieldops-photos-v1)",
            "Text + image",
            "v0 stub",
        ],
        [
            "Call Center",
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset (mirrored) + ligaments/callcenter-multiling-v1",
            "Text",
            "Bitext live; multiling v0 stub",
        ],
    ]
    cells = [rows[0]] + [[p(x, "Body") for x in r] for r in rows[1:]]
    story.append(header_table(cells, col_widths=[26 * mm, 70 * mm, 32 * mm, 42 * mm]))

    story.append(p("9.1 Compliance and PII", "H2"))
    for line in [
        "ITR / GST data flowing through the agent is run through a redaction pass before any HF push.",
        "Customer-uploaded datasets default to <b>private</b> HF repos, owned by the customer&rsquo;s tenant org.",
        "Audit log of every tool call (esp. <font name='Courier'>hf_repo_files</font>, <font name='Courier'>hf_jobs</font>) for regulated pillars.",
        "Region-pin: customer can require HF Jobs hardware in a specific region for data-residency compliance.",
    ]:
        story.append(bullet(line))


def roadmap(story):
    story.append(PageBreak())
    story.append(p("10. Roadmap", "H1"))
    rows = [["Phase", "Theme", "Key deliverables", "Exit criteria"]]
    rows += [
        [
            "Phase 0 (now)",
            "Foundation",
            "Agent loop stable; backend on 7860; Bitext-telco recipe; HF Spaces deploy.",
            "End-to-end demo works for at least one pillar.",
        ],
        [
            "Phase 1",
            "Pillar specialisation",
            "Pillar selector UI; pillar-conditioned prompts; per-pillar dataset &amp; model catalogues; CC1-1, ITR-1, GST-1, FO-1.",
            "All four pillars have a working &lsquo;hello world&rsquo; finetune.",
        ],
        [
            "Phase 2",
            "Compliance &amp; multilingual",
            "PII redaction; Indian-language defaults; per-pillar eval harness; quotas hardened.",
            "ITR &amp; GST flows pass an internal compliance review.",
        ],
        [
            "Phase 3",
            "MultiCloud GA",
            "AWS + Azure reference deployments; tenant-scoped HF orgs; customer-cloud serving recipes.",
            "Two pilot customers running on two different clouds.",
        ],
        [
            "Phase 4",
            "Vertical depth",
            "Structured-output finetunes (ITR XML, GSTR JSON); VLM finetunes for FieldOps photos; agent-assist real-time path for Call Center.",
            "At least one paid production deployment per pillar.",
        ],
    ]
    cells = [rows[0]] + [[p(x, "Body") for x in r] for r in rows[1:]]
    story.append(header_table(cells, col_widths=[28 * mm, 32 * mm, 65 * mm, 45 * mm]))


def metrics(story):
    story.append(PageBreak())
    story.append(p("11. Success Metrics", "H1"))
    story.append(p("11.1 Product metrics", "H2"))
    for line in [
        "<b>Time-to-first-finetune</b> per pillar &lt; 30 minutes from sign-in to a pushed model.",
        "<b>Agent success rate</b>: % of sessions reaching <font name='Courier'>turn_complete</font> with a pushed model &ge; 80% on canonical prompts.",
        "<b>Per-pillar eval score</b>: each pillar exposes a leaderboard-grade benchmark; production checkpoints must beat the base model by &ge;5 pts.",
        "<b>Multi-cloud parity</b>: same demo runs identically on HF Spaces, GCP, AWS, Azure.",
    ]:
        story.append(bullet(line))
    story.append(p("11.2 Business metrics", "H2"))
    for line in [
        "Active tenants per pillar.",
        "Finetuned models pushed per tenant per month.",
        "Tokens served via the platform per tenant per month.",
        "Pilot conversions &mdash; pilots to paid contracts.",
    ]:
        story.append(bullet(line))


def risks(story):
    story.append(p("12. Risks and Mitigations", "H1"))
    rows = [["Risk", "Likelihood", "Impact", "Mitigation"]]
    rows += [
        [
            "Generic ML platforms (AWS Bedrock, GCP Vertex) absorb the use cases",
            "High",
            "High",
            "Compete on Indian-domain depth: curated pillar datasets, eval suites, and tax-/regulation-aware prompts they cannot match.",
        ],
        [
            "PII leakage on ITR / GST flows",
            "Medium",
            "Critical",
            "Mandatory redaction pass; private HF repos; region-pinned training; audit logs; signed data-processing addendum for pilots.",
        ],
        [
            "HF Jobs availability or pricing shifts",
            "Medium",
            "High",
            "Keep training scripts cloud-agnostic; have AWS SageMaker and Azure AML fallback recipes ready in Phase 3.",
        ],
        [
            "Agent doom-loops on hard prompts inflating cost",
            "Medium",
            "Medium",
            "Doom-loop detector already in place; add per-tenant token quota cap and per-pillar guard prompts.",
        ],
        [
            "Indian-language datasets are sparse or low quality",
            "High",
            "Medium",
            "Curate in-house; partner with annotation vendors; use Indic-tuned base models (IndicGemma, Sarvam, AI4Bharat).",
        ],
        [
            "Vendor lock-in to Hugging Face Hub",
            "Low",
            "Medium",
            "Model artifacts are portable (safetensors + tokenizer); abstract Hub access behind a thin wrapper to allow alternative registries later.",
        ],
    ]
    cells = [rows[0]] + [[p(x, "Body") for x in r] for r in rows[1:]]
    story.append(header_table(cells, col_widths=[55 * mm, 22 * mm, 18 * mm, 75 * mm]))


def open_questions(story):
    story.append(PageBreak())
    story.append(p("13. Open Questions / Decisions Needed", "H1"))
    for line in [
        "<b>Tenant model</b>: one HF org per customer vs. shared <font name='Courier'>ligaments-tenants</font> org with sub-namespaces &mdash; pick by Phase 1.",
        "<b>Billing</b>: pass-through HF Jobs cost vs. all-in subscription &mdash; needs commercial input before Phase 3.",
        "<b>Indic-LLM partnership</b>: lean on AI4Bharat / Sarvam open weights, or co-train? &mdash; decide before Phase 2.",
        "<b>Frontend framework</b>: IMPLEMENTATION.md proposes a single static HTML to simplify Docker; confirm we won&rsquo;t need React for the pillar-rich UI long-term.",
        "<b>Eval harness shape</b>: build per-pillar from scratch vs. wrap <font name='Courier'>lm-eval-harness</font> &mdash; decide in Phase 2.",
        "<b>On-prem story</b>: do we ship a K8s Helm chart in Phase 4, or hold for Phase 5?",
    ]:
        story.append(bullet(line))


def appendix(story):
    story.append(PageBreak())
    story.append(p("Appendix A &mdash; Repository layout", "H1"))
    story.append(
        code(
            "huggingface-ml-intern-finetuning/\n"
            "├── agent/                 # autonomous agent runtime (installed editable as hf-agent)\n"
            "│   ├── core/              # agent_loop, handlers, session, context_manager\n"
            "│   ├── tools/             # hf_*, sandbox, github, planning, mcp\n"
            "│   ├── prompts/           # system + pillar-conditioned prompts (to be added)\n"
            "│   ├── config.py\n"
            "│   └── main.py            # CLI entrypoint (`ml-intern`)\n"
            "├── backend/               # FastAPI app on port 7860\n"
            "│   ├── main.py            # uvicorn entry; mounts static/ + routers\n"
            "│   ├── routes/            # agent.py, platform.py (planned), auth\n"
            "│   ├── session_manager.py\n"
            "│   ├── user_quotas.py\n"
            "│   └── start.sh           # HF Spaces entrypoint\n"
            "├── frontend/              # React + Vite (being replaced by backend/static/index.html)\n"
            "├── configs/               # main_agent_config.json, model defaults\n"
            "├── train_gemma_telco.py   # reference Call-Center finetune recipe\n"
            "├── tests/\n"
            "├── Dockerfile             # HF Spaces / Cloud Run compatible\n"
            "├── pyproject.toml         # hf-agent package metadata\n"
            "└── README.md / IMPLEMENTATION.md / Mandatory APIs.md / All APIs.md\n"
        )
    )

    story.append(p("Appendix B &mdash; Backend run command", "H1"))
    story.append(
        code(
            "cd D:\\_AI_\\LigaX_ML_Intern\\huggingface-ml-intern-finetuning\n"
            "pip install -e .              # first time only, or after pyproject changes\n"
            "cd backend\n"
            "uvicorn main:app --host 0.0.0.0 --port 7860 --reload\n"
        )
    )
    story.append(
        p(
            "Gotcha: if <font name='Courier'>agent.config</font> fails to import, check "
            "<font name='Courier'>C:\\Users\\Tushar Das\\.conda\\envs\\myenv\\Lib\\site-packages\\</font> for a "
            "leftover <font name='Courier'>_editable_impl_ml_agent.pth</font> from the older "
            "<font name='Courier'>D:\\_AI_\\Liga_ML_Intern\\ml-agent</font> repo and delete it before debugging further.",
            "BodyMuted",
        )
    )

    story.append(p("Appendix C &mdash; Version history", "H1"))
    rows = [["Version", "Date", "Notes"]]
    rows += [
        [
            "1.0",
            "2026-05 (earlier)",
            "Parity gap closure between ml-intern and huggingface-ml-intern-finetuning.",
        ],
        [
            "2.0",
            date.today().isoformat(),
            "Reframed around MultiCloud ultimate goal: ITR, GST, FieldOps, Call Center. Supersedes v1.0 for product scope; v1.0 still valid for parity engineering tasks.",
        ],
    ]
    cells = [rows[0]] + [[p(x, "Body") for x in r] for r in rows[1:]]
    story.append(header_table(cells, col_widths=[20 * mm, 30 * mm, 120 * mm]))


# ---------------- Build ----------------


def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title="LigaX ML Intern — PRD v2.0",
        author="Tushar Das",
    )
    story = []
    cover(story)
    executive_summary(story)
    vision(story)
    domains(story)
    personas(story)
    current_state(story)
    goals_nongoals(story)
    functional_requirements(story)
    architecture(story)
    data_strategy(story)
    roadmap(story)
    metrics(story)
    risks(story)
    open_questions(story)
    appendix(story)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
