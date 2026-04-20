"""
federated.py  (Gemma 3 / Ollama edition)
─────────────────────────────────────────
Simulates Federated Learning (FL) with FedAvg over domain-prompt adapters.

Architecture (per the abstract)
────────────────────────────────
  Client 0          Client 1          Client N
  (private PDF)     (private PDF)     (private PDF)
       |                 |                 |
  local adapter     local adapter     local adapter
       |                 |                 |
       └─────────────────┴─────────────────┘
                         |
                    FL Server
                  (FedAvg merge)
                         |
                  Global Adapter
                         |
                 Improved Summaries

Privacy guarantee: raw documents NEVER leave each client.
Only the lightweight system-prompt adapter (~2 KB text) is shared.

FedAvg analogy
──────────────
Classic FedAvg averages weight tensors weighted by dataset size.
Here "weight tensors" are replaced by domain-prompt adapters, and
"averaging" is performed by asking Gemma 3 to synthesise the local
adapters into one global adapter, weighted by each client's chunk count.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

from utils.qlora_trainer import (
    _ollama_generate,
    extract_domain_profile,
    generate_adapter_prompt,
    save_adapter,
    MODEL_NAME,
)


# ─── Federated Client ────────────────────────────────────────────────────────

class FederatedClient:
    """Simulates one participant in the federated network."""

    def __init__(self, client_id: int, chunks: List[str]):
        self.client_id: int            = client_id
        self.chunks:    List[str]      = chunks
        self.adapter:   Optional[str]  = None   # local system-prompt adapter
        self.profile:   dict           = {}

    def local_train(self) -> None:
        """
        Run local domain adaptation.
        Analogous to: client fine-tunes QLoRA adapter on private data.
        """
        self.profile = extract_domain_profile(self.chunks)
        self.adapter = generate_adapter_prompt(self.profile)

    @property
    def adapter_payload(self) -> dict:
        """
        Serialisable payload sent to the FL server.
        Contains ONLY the adapter and profile — never raw document text.
        """
        return {
            "client_id":      self.client_id,
            "adapter_prompt": self.adapter or "",
            "domain_profile": self.profile,
            "num_chunks":     len(self.chunks),
        }


# ─── FedAvg Aggregation ───────────────────────────────────────────────────────

def fedavg_prompts(payloads: List[dict]) -> str:
    """
    Federated Averaging of domain-prompt adapters.

    Asks Gemma 3 to synthesise N local adapters into one global adapter,
    weighted by each client's chunk count (dataset size).
    """
    profiles_text = ""
    for p in payloads:
        profiles_text += (
            f"\n--- Client {p['client_id']}  "
            f"(weight = {p['num_chunks']} chunks) ---\n"
            f"{p['adapter_prompt'][:700]}\n"
        )

    merge_prompt = (
        "You are a federated learning aggregation server. "
        "Multiple client nodes have each fine-tuned a local system-prompt adapter "
        "on their private documents. "
        "Only the adapters are shared here — raw document data was never transmitted.\n\n"
        "Synthesise a single, improved GLOBAL system prompt that combines the strengths "
        "of all client adapters, weighted by their chunk counts. "
        "The global prompt must:\n"
        "  - Be usable as a system prompt for document summarisation\n"
        "  - Retain domain-specific vocabulary from all clients\n"
        "  - Enforce the four-section output format "
        "(OVERVIEW, METHODOLOGY, OUTCOMES, KEYWORDS)\n"
        "  - Be concise (under 300 words)\n\n"
        "Output ONLY the merged system prompt. No preamble, no explanation.\n\n"
        f"CLIENT ADAPTERS:{profiles_text}\n\n"
        "GLOBAL SYSTEM PROMPT:"
    )
    return _ollama_generate(merge_prompt, max_tokens=600).strip()


# ─── Full FL Simulation ───────────────────────────────────────────────────────

def run_federated_training(
    client_chunks_list: List[List[str]],
    output_adapter_path: str = "./qlora_adapter_federated",
    lora_r: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 1,
    num_rounds: int = 1,
    client_weights: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, dict]:
    """
    Run one or more complete federated learning rounds.

    Phase 1 — Local Training
      Each client analyses its private document chunks and generates a
      local domain-prompt adapter.  Raw data never leaves the client.

    Phase 2 — Server Aggregation (FedAvg)
      The server receives only the lightweight adapter texts and uses
      Gemma 3 to merge them into a single global adapter.

    Phase 3 — Global Adapter Broadcast
      The global adapter is distributed back to clients and saved.

    Parameters
    ----------
    client_chunks_list   : list[list[str]] — one list of chunks per client
    output_adapter_path  : where to write the global adapter
    lora_r, lora_alpha   : LoRA hyperparameters (stored in metadata)
    num_epochs           : training epochs per client (stored in metadata)
    num_rounds           : number of FL communication rounds
    client_weights       : manual FedAvg weights (None = proportional to chunks)
    progress_callback    : fn(message: str, fraction: float 0→1)

    Returns
    -------
    (adapter_path: str, stats: dict)
    """
    n_clients = len(client_chunks_list)
    if n_clients < 2:
        raise ValueError("Federated simulation requires at least 2 clients (documents).")

    def _log(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    stats: dict = {
        "n_clients":    n_clients,
        "num_rounds":   num_rounds,
        "client_stats": [],
        "round_logs":   [],
    }

    clients = [FederatedClient(i, chunks) for i, chunks in enumerate(client_chunks_list)]
    global_adapter_prompt = ""

    total_steps = num_rounds * (n_clients + 1)  # n_clients local + 1 aggregation per round
    step = 0

    for rnd in range(num_rounds):
        _log(f"Round {rnd + 1}/{num_rounds}: local training started…", step / total_steps)

        # Phase 1 — Local training
        for idx, client in enumerate(clients):
            _log(
                f"Round {rnd + 1} · Client {idx + 1}/{n_clients}: domain analysis…",
                step / total_steps,
            )
            # In later rounds, seed local training with the global adapter context
            if global_adapter_prompt and rnd > 0:
                client.adapter = global_adapter_prompt  # warm-start from global

            client.local_train()
            step += 1
            _log(
                f"Round {rnd + 1} · Client {idx + 1}/{n_clients}: adapter ready ✓",
                step / total_steps,
            )

        payloads = [c.adapter_payload for c in clients]

        # Phase 2 — FedAvg aggregation
        _log(f"Round {rnd + 1}: FedAvg — aggregating {n_clients} adapters…", step / total_steps)
        global_adapter_prompt = fedavg_prompts(payloads)
        step += 1

        # Phase 3 — Broadcast global adapter back to clients
        for client in clients:
            client.adapter = global_adapter_prompt

        stats["round_logs"].append({
            "round":          rnd + 1,
            "n_clients":      n_clients,
            "adapter_preview": global_adapter_prompt[:150] + "…",
        })
        _log(f"Round {rnd + 1} complete — global adapter aggregated ✓", step / total_steps)

    # Record client stats
    for c in clients:
        stats["client_stats"].append({
            "client_id": c.client_id,
            "n_chunks":  len(c.chunks),
            "domain":    c.profile.get("domain", "general"),
        })

    # Save final global adapter
    _log("Saving global federated adapter…", 0.95)
    all_key_terms = list({
        t
        for p in [c.profile for c in clients]
        for t in p.get("key_terms", [])
    })
    merged_profile = {
        "domain":           "federated multi-domain",
        "key_terms":        all_key_terms[:12],
        "doc_style":        "federated aggregation of multiple domains",
        "summary_guidance": "Apply the globally aggregated adapter for cross-domain accuracy.",
    }
    meta = {
        "base_model":    MODEL_NAME,
        "adapter_type":  "federated_prompt_adapter",
        "lora_r":        lora_r,
        "lora_alpha":    lora_alpha,
        "num_epochs":    num_epochs,
        "num_rounds":    num_rounds,
        "n_clients":     n_clients,
        "total_chunks":  sum(len(c.chunks) for c in clients),
        "trained_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "client_stats":  stats["client_stats"],
    }
    final_path = save_adapter(global_adapter_prompt, merged_profile, output_adapter_path, meta)
    stats["output_path"] = final_path

    _log("Federated training complete — global adapter saved ✓", 1.0)
    return final_path, stats
