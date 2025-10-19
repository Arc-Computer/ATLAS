
# Foundational Paper Outline: ATLAS Core

**Title:** ATLAS Core: A Continual Learning Engine for Building Agentic World Models

**Abstract:**
The prevailing paradigm of architecturally static AI agents, which cannot learn from experience, fundamentally limits their reliability and return on investment. This paper introduces ATLAS, an open-source, closed-loop architecture that enables teams to build **domain-specialized World Models** of their operational environments. Our core innovation is a symbiotic **hybrid learning engine**. An **online adaptive runtime**, featuring a Teacher-Student orchestration, captures a rich, structured dataset of the agent's action space. This unique dataset of structured experience then feeds an **offline GRPO training engine** that continually improves the Teacher model. This virtuous cycle, powered by our SOTA Reward Interpretation Model (RIM), resolves the Stability-Plasticity Dilemma and produces compounding gains in performance and efficiency. We instantiate ATLAS on **ArcOps-Cyber**, a reproducible slice of ExCyTIn-Bench security incidents, and show that efficiency gains observed at runtime are a direct, measurable result of the Teacher’s weight updates. We argue that our full, closed-loop system, not any single algorithmic component, constitutes the defensible moat for building truly adaptive agents and discuss how the same pattern generalizes to additional verticals.

---

## 1. Introduction: The End of Static Agents

*   **Narrative Point:** Today's powerful LLMs are still architecturally static, preventing them from learning from experience. This manifests as both a performance ceiling and a significant inefficiency, as static agents must expend considerable compute reasoning through problems they have solved before. The next frontier is building agents that can continually learn and construct a "World Model" of their environment.

*   **The Alignment Paradox:** Recent work has highlighted a critical paradox: while the need for continually learning models is clear, our methods for ensuring their safety and alignment are built for a static world. As Chan (2025) notes, when models become *dynamic*, our existing evaluation and alignment techniques "go out the window." This presents a major impediment to the deployment of truly adaptive agents. In this paper, we introduce ATLAS as an architecture designed from the ground up to solve this paradox. ATLAS is not just a continual learning technique; it is a complete system that integrates **dynamic skill acquisition with continual evaluation**. Our online adaptive runtime, powered by our SOTA reward model, provides the "cheap, fast evaluation" necessary to steer the agent in real-time, while our offline training engine provides the stability to "maintain existing alignment." We demonstrate that by solving the architectural problem, we also provide a practical solution to the evaluation problem, paving the way for the safe and reliable deployment of agents that truly learn.

*   **Key Concepts:** World Model, Continual Learning (CL), Dynamic Skill Acquisition, Continual Evaluation, Stability-Plasticity Dilemma.
*   **Scope (this paper):** Evidence is grounded in ArcOps-Cyber (ExCyTIn-derived) incident response tasks. Multi-domain generalization is reserved for future work.
*   **Supporting Asset:** Figure 1: A conceptual diagram showing a static agent's performance plateauing over time, contrasted with a continually learning agent's compounding "J-curve."

## 2. The ATLAS Architecture: A Hybrid System for Continual Learning

*   **Narrative Point:** To build a World Model, we need a system that can both learn foundational knowledge and adapt to new information. This section introduces ATLAS's hybrid architecture as a holistic solution. A key design principle is that the **Student** is the user's existing, model-agnostic agent. ATLAS enhances this Student via external guidance without modifying its weights at runtime, acting as a "drop-in" learning harness.
*   **Key Concepts:** Teacher-Student Model, Model-Agnostic Student, Hybrid Learning, Online Data Generation, Offline Training.
*   **Supporting Asset:** Figure 2: A high-level architectural diagram of the full ATLAS loop, explicitly labeling the "Online Adaptive Runtime" (Data Generation) and "Offline Training Engine" (Knowledge Distillation) components.

### 2.1. Defining the Agentic World Model
*   **Narrative Point:** We clarify that the World Model is not a separately trained artifact but an **emergent property** of the ATLAS learning process. It is the combination of: (1) The foundational, generalizable reasoning structures encoded in the **Teacher model's weights** through offline GRPO; and (2) The dynamic, explicit history of action-outcome pairs representing procedural knowledge stored in **persistent memory**. The system *builds* the World Model; it does not train it directly.
*   **Key Concepts:** Emergent Property, Foundational Reasoning, Procedural Knowledge, Persistent Memory.

## 3. The Data Engine: Capturing the Action Space as Raw Material

*   **Narrative Point:** A World Model must be built from data that reflects the environment's dynamics. This section reframes the Teacher-Student interaction not as the end-goal, but as a sophisticated data generation mechanism. Every interaction is a structured training asset that captures the agent's "action space," the raw material for the World Model.
*   **Key Concepts:** Action Space, Structured Training Asset, Persistent Memory, Telemetry.
*   **Supporting Asset:** Figure 3: A diagram illustrating a single interaction (Task -> Student Attempt -> Teacher Guidance -> Final Outcome -> Reward Signal) and how it is transformed into a structured JSONL record for the training pipeline.

## 4. The Learning Engine: A Symbiotic Online-Offline Loop

*   **Narrative Point:** This section details how ATLAS's dual-mode learning engine solves the Stability-Plasticity Dilemma through a symbiotic relationship between online data generation and offline training.
*   **4.1. Online Adaptive Runtime: Generating High-Fidelity Experience.**
    *   **Narrative:** The `atlas-sdk` runtime acts as a superior data-generation engine. Its capability probe and adaptive supervision lanes (`auto`, `paired`, `coach`, `escalate`) function as a real-time curriculum, ensuring that the captured experience is both relevant and rich in learning signal. This provides the "Plasticity" for the system.
*   **4.2. Offline GRPO Engine: Distilling Experience into a Stable World Model.**
    *   **Narrative:** The `ATLAS` core trainer takes the high-fidelity data from the runtime and uses GRPO to distill this experience into the Teacher model's weights. This process builds the stable, foundational layer of the World Model, providing long-term "Stability."
*   **Key Concepts:** Data Flywheel, GRPO, Adaptive Runtime, Stability-Plasticity Dilemma.
*   **Supporting Assets:**
    *   Figure 4: The "J-Curve of Learning" chart, demonstrating compounding runtime improvement, framed as a result of the World Model becoming more accurate.
    *   Table 1: The performance metric: "+165% performance gain in under two hours for less than $10."

## 5. The Reward System (RIM): A High-Fidelity Signal for the Flywheel

*   **Narrative Point:** An accurate World Model requires an accurate feedback signal. This section positions your SOTA reward system as the crucial component that powers the entire data flywheel, ensuring that the experience captured online is correctly labeled for effective offline training.
*   **Key Concepts:** Reward Interpretation Model (RIM), Ensemble of Judges.
*   **Supporting Asset:** Figure 5: The RewardBench V2 leaderboard chart, highlighting the 93.7% SOTA accuracy.

## 6. Experimental Validation

This section provides empirical evidence for the ATLAS architecture using the **ArcOps-Cyber** benchmark—a curated subset of ExCyTIn-Bench security incidents. The goal is to demonstrate domain-specialized world models that deliver immediate runtime efficiency with off-the-shelf models, retain gains when memory is disabled, and continue improving after offline GRPO. We reserve broader cross-domain generalization for future work and track it explicitly in the roadmap.

**SecRL Alignment (Day 1 Update):** The runtime now mirrors Microsoft's SecRL ArcOps-Cyber protocol end-to-end. Student runs issue parameterised queries against the unpacked SecRL MySQL logs via a dedicated `secrl_sql` tool, and `adaptive_teaching.reward` wraps the upstream evaluator prompts to report the authoritative SecRL reward alongside our RIM telemetry. Batch experiments therefore expose the same success metric (SecRL reward) used in the original paper while retaining ATLAS-specific efficiency diagnostics (token, latency, RIM guidance curves).

### 6.1. Runtime Value: Efficiency and Accessibility within ArcOps-Cyber

*   **Narrative Point:** We first prove the immediate, out-of-the-box value of the ATLAS adaptive runtime on ArcOps-Cyber incidents. A user can pair any two standard models (e.g., Llama-3.2-8B Student + Claude/GPT-4o Teacher) via the atlas-sdk harness and achieve higher success rates with greater overall system efficiency while handling realistic threat-investigation questions.
*   **Key Concepts:** Model-Agnostic Harness, Off-the-Shelf Pairings, Orchestration Framework, System-Level Efficiency, J-Curve of Learning.
*   **Supporting Asset:** Table 2 & Figure 6: ArcOps-Cyber runtime J-curve and iso-success efficiency frontier (Student-only vs Student+Teacher).

### 6.2. Architectural Value: Specialized Teacher & Closed Loop on ArcOps-Cyber

*   **Narrative Point:** After establishing the framework's value, we prove that Arc's specialized technology provides a distinct advantage on the same domain. This is a two-part experiment.
*   **Part A (Ablation Study):** Replace the off-the-shelf Teacher from 6.1 with our GRPO-specialized `Teacher_v0`. Show improved pedagogical efficiency (success per guidance token, latency reduction).
*   **Part B (Generational Improvement):** Use ArcOps-Cyber traces generated by `Teacher_v0` to fine-tune `Teacher_v1`. Demonstrate that `Teacher_v1` verifiably outperforms `Teacher_v0` on held-out ArcOps-Cyber tasks and retains gains when persistent memory is disabled.
*   **Key Concepts:** Pedagogical Efficiency, Compounding Intelligence, Closed-Loop Learning, Generational Improvement.
*   **Supporting Assets:**
    *   Table 3: Guidance efficiency comparison (Teacher baseline vs `Teacher_v0`).
    *   Figure 7: `Teacher_v0` vs `Teacher_v1` performance on ArcOps-Cyber, including memory-off controls.
    *   Appendix Figure: SecRL SQL telemetry captured through the `secrl_sql` adapter (query latency, incident coverage, evidence citations).

## 7. Related Work

*   **Narrative Point:** Position ATLAS within the broader AI landscape by comparing it to contemporary research threads. We show that ATLAS integrates the best ideas from these threads into a single, more powerful architecture. Our primary differentiation lies not in a single algorithm, but in the design of the complete, closed-loop data flywheel.
*   **Supporting Asset:** Table 4: A summary table comparing ATLAS to related works across key dimensions (e.g., Core Architecture, Learning Mechanism, Primary Artifact).

### 7.1. vs. Experience-Based Learning (e.g., "Agent Learning via Early Experience", "ReasoningBank")
*   **Narrative:** We share the core thesis that agents should learn from their own successes and failures. However, ATLAS's dual-agent architecture with a specialized Teacher and offline GRPO training provides a more robust learning signal and a more durable learning mechanism than single-agent self-reflection or memory-only updates.

### 7.2. vs. Context Engineering (e.g., "Agentic Context Engineering")
*   **Narrative:** We agree that evolving context is critical. However, where ACE focuses on engineering the prompt, ATLAS focuses on training the Teacher model itself. This allows for deeper, more nuanced learning that is captured in the model's weights, not just its input text.

### 7.3. vs. Parametric Memory (e.g., "Hierarchical Memories Pretraining")
*   **Narrative:** We agree on the value of separating reasoning from knowledge. However, their approach focuses on storing declarative knowledge ("what") in a static memory bank. ATLAS focuses on capturing procedural, experiential knowledge ("how") from a live environment, which is more relevant for agentic tasks.

### 7.4. vs. Scalable RL Recipes (e.g., "The Art of Scaling RL")
*   **Narrative:** We acknowledge that RL training pipelines are becoming a predictable, engineered science. Our moat is not the GRPO algorithm itself, but the proprietary, high-signal dataset generated by our unique online runtime, which makes our training process fundamentally more effective.

### 7.5. vs. Foundational World Models (e.g., CWM)
*   **Narrative:** We differentiate our approach from the training of large, monolithic foundational models that are pre-trained on execution traces (e.g., Meta's CWM). CWM learns the universal semantics of code execution ("what code does"). ATLAS is a complementary, higher-level framework that learns the procedural strategy required to solve a specific task in a live environment ("how to achieve a goal"). ATLAS is a continual learning system that can wrap *any* base model (including one like CWM) to make it adaptive to a specific operational context.

## 8. Limitations and Broader Impact

*   **Narrative Point:** We honestly discuss the limitations of our approach and the broader implications of autonomous learning systems.
*   **Limitations:** This section will address challenges such as operating in noisy, low-signal environments; the dependency on high-quality telemetry; and the potential for the World Model to learn spurious or incorrect environmental dynamics.
*   **Broader Impact:** We discuss the ethical considerations and necessary guardrails for deploying agents that can learn and adapt autonomously in production, including strategies for monitoring, auditing, and preventing undesirable emergent behaviors.

## 9. Conclusion & Future Work

*   **Narrative Point:** Conclude by re-asserting that the future of AI lies in building compounding intelligence assets. ATLAS provides the first practical, open-source engine for constructing **domain-specialized** agentic World Models. We synthesize our ArcOps-Cyber results to argue that the observable gains in system efficiency are the direct result of the Teacher’s weight updates and procedural memory. As an agent's understanding of a specific operational domain becomes more accurate, its actions become more precise, requiring less exploratory reasoning and thus consuming fewer resources. **Efficiency, therefore, is the shadow cast by intelligence.**
*   **Future Work:** Extend the flywheel to additional domains (SRE, supply chain, identity), explore richer forms of memory, and complete the live ExCyTIn environment integration—moving stepwise toward generalized world models while maintaining the system-level guarantees demonstrated here.
