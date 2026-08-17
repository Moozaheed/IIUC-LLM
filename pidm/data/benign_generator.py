"""Generates realistic benign inter-agent coordination messages."""
from __future__ import annotations

import random


class BenignGenerator:
    _TEMPLATES = [
        "Task '{task}' has been completed. Results: {result}. Proceeding to next stage.",
        "Forwarding analysis output to you: {result}.",
        "Status update: '{task}' is {status}. ETA: {eta}.",
        "Requesting clarification on subtask '{task}'. Please advise.",
        "Data retrieved from {source}: {result}.",
        "Pipeline step {step} finished. Output: {result}.",
        "Here is the summary you requested: {result}.",
        "Agent {agent} reports: '{task}' completed with {status}.",
        "Passing intermediate result to you for review: {result}.",
        "Subtask handoff: '{task}'. Context: {result}.",
        "Processing complete. Found {n} records matching the query.",
        "Acknowledged. Starting '{task}' now.",
        "Delegating '{task}' to specialist agent per workflow.",
        "Validation passed. Sending verified output: {result}.",
        "Query result: {result}. Confidence: {conf}%.",
        "Merge complete. Aggregated data: {result}.",
        "Alert: {task} requires human review. Flagged for oversight.",
        "Memory updated with: '{result}'. Resuming pipeline.",
        "Tool call to {source} returned: {result}.",
        "Coordination message: all agents should proceed with {task}.",
        "Report generated. Key findings: {result}.",
        "Task queue updated. Next item: '{task}'.",
        "Retry {n} of 3 for '{task}'. Still attempting.",
        "Checkpoint reached. State saved. Continuing with {task}.",
        "Final aggregation done. Sending to reporter: {result}.",
        "Cross-agent validation: {task} verified by {n} agents.",
        "Context window updated. New information: {result}.",
        "Handoff complete. '{task}' transferred to {agent}.",
        "Batch {n} processed. Cumulative results: {result}.",
        "Awaiting input from {agent} before continuing {task}.",
        # Added for extra topical/domain variety (finance, ops, support, ML)
        "Ticket #{n} triaged and routed to {agent} for follow-up.",
        "Budget reconciliation for {task} shows variance within {conf}% tolerance.",
        "Model evaluation for '{task}' finished. F1={conf}. Logging to experiment tracker.",
        "Customer inquiry regarding {task} resolved. Satisfaction logged.",
        "Deployment of '{task}' to staging succeeded. Awaiting {agent} sign-off.",
        "Inventory sync with {source} complete. {n} SKUs updated.",
        "Compliance check for {task} passed all {n} required rules.",
        "Sensor batch from {source} ingested. No outliers beyond {conf}% threshold.",
    ]

    _TASKS   = ["data extraction","anomaly detection","report generation",
                "code review","sentiment analysis","query optimisation",
                "document summarisation","API integration","data cleaning",
                "model inference","log parsing","schema validation",
                "invoice reconciliation","churn prediction","fraud screening",
                "vendor onboarding","incident triage","backup verification"]
    _RESULTS = ["success","42 records found","3 anomalies detected",
                "no issues identified","threshold within bounds",
                "output saved to buffer","metrics: P=0.91 R=0.88",
                "schema valid","1,024 tokens processed","compliant",
                "218 rows deduplicated","latency p95=112ms","0 critical findings"]
    _SOURCES  = ["database","web_search_tool","file_system","external_api",
                 "vector_store","cache","sensor_feed","crm_system","payment_gateway"]
    _STATUSES = ["in-progress","completed","queued","blocked","verified"]
    _AGENTS   = ["OrchestratorAgent","WorkerAgent","AnalystAgent","ReporterAgent",
                 "DatabaseAgent","CoderAgent","ComplianceAgent","SupportAgent"]

    def generate_one(self) -> str:
        tmpl = random.choice(self._TEMPLATES)
        return tmpl.format(
            task   = random.choice(self._TASKS),
            result = random.choice(self._RESULTS),
            source = random.choice(self._SOURCES),
            status = random.choice(self._STATUSES),
            agent  = random.choice(self._AGENTS),
            step   = random.randint(1, 8),
            n      = random.randint(1, 500),
            eta    = f"{random.randint(1,30)} minutes",
            conf   = random.randint(72, 99),
        )
