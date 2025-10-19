# Arc: Pre-Seed Investment Memo

<aside>
💡

## Executive Summary

Today's AI agents are architecturally static, capping the ROI on enterprise AI’s projected [$20B+](https://menlovc.com/perspective/2025-mid-year-llm-market-update/) in API spend. An agent’s inability to learn from its environment prevents it from acquiring skill from experience, blocking companies from building their true goal: a proprietary, compounding intelligence asset.

Arc provides the essential data engine for this new paradigm, offering the end-to-end infrastructure for any enterprise to build its own proprietary World Model. 

By turning a company's live operational environment into the ultimate training ground, our system allows agents to learn, simulate outcomes, and adapt in real-time.

Our core innovation is a hybrid learning architecture powered by our proprietary reward system, which recently set a [**new state-of-the-art (SOTA) on the RewardBench V2 benchmark](https://www.arc.computer/blog/ATLAS-Reward-System).** We can take a customer's existing agent and deliver:

- A **+165% performance gain**
- In **under 2 hours** of live adaptation
- For **less than $10** in compute costs

This allows customers in high-stakes domains like cybersecurity, SRE, and supply chain to achieve SOTA performance while significantly driving down their token costs. We are proving this with tangible metrics through real-world benchmark collaborations with industry leaders like **Datadog, Auth0 (Okta), and Reply.** Our infrastructure is designed to be embedded, creating a deep, technical moat that is nearly impossible to rip out.

Led by a team combining a PhD in Learning Science, a PhD in first-principles Neuroscience, and proven enterprise platform GTM, we are building the definitive standard for how AI systems learn and acquire skill. 

We are raising a $3M pre-seed to scale with our initial design partners and capture this foundational new category.

**Key Links → [Website](https://www.arc.computer/) | [Github](https://github.com/Arc-Computer/ATLAS)  | [Docs](https://docs.arc.computer/) | [HuggingFace |](https://huggingface.co/Arc-Intelligence) [Research](https://www.arc.computer/research) | [Linkedin](https://www.linkedin.com/company/arc-eval/) | [Twitter](https://x.com/IntelligenceArc)**

**Questions? See our [Investor FAQ](https://www.notion.so/Investor-FAQ-272778a1210a80fcac08f3d6ae3a9e6b?pvs=21) for deep-dive responses to each section.**

</aside>

## Core Technical Concepts

1. **What a "World Model" is:** A predictive model of an environment’s latent dynamics, the “how the world works” layer. It is the core component that enables an agent to perform complex planning, simulate potential outcomes, and estimate uncertainty before taking a costly action in the real world.
2. **What Continual Learning (CL) is:** The capability for an ML system to **keep learning after deployment** from a non-stationary (constantly changing) stream of data **without forgetting** prior capabilities. This is the process that transforms a static AI tool into a compounding, appreciating asset that adapts to its environment.
3. **The Role and Limits of Reinforcement Learning (RL):** In post-training, RL is a powerful technique where a model learns optimal behavior by interacting with an environment to maximize a reward signal. However, its architectural reliance on large, computationally expensive batch updates makes it ill-suited for the small, frequent, and highly parameter-efficient updates required by true Continual Learning from a live data stream.
4. **The Stability-Plasticity Dilemma:** A fundamental trade-off in continual learning. While storing past experiences provides model **stability** (mitigating catastrophic forgetting), it can reduce **plasticity**, making the model biased toward prior tasks and less able to learn new ones. Effectively navigating this dilemma requires a system designed to manage both. The architectural approach is to decouple these two concerns: a stable, persistent knowledge base (the World Model) provides the foundation for stability, while a dynamic learning engine continually introduces new skills, ensuring the system retains its plasticity.
****

## **The Market Shift**

> In complex adaptive systems, the geometry of interactions is not static but perpetually evolving. By the time a model has been trained to map one configuration, the system itself has already reconfigured into another.
> 

This market reality is forcing a compression of the AI stack. As foundational models commoditize and inference costs plummet, durable value is rebundling away from the *mechanisms* of intelligence (the model) and toward **the system** that governs how intelligence improves. **The system is the product.**

This represents a fundamental shift: moving beyond models trained on the *syntax* of static code to models that learn the *semantics* of live execution. The most durable moat is not the model itself, but the engine that captures proprietary operational data and uses it to build a true World Model. The new strategic imperative for the enterprise is no longer just using AI, but building this defensible, organization-specific intelligence asset.

This creates a new reality for the enterprise, defined by four core trends:

1. **The New Moat is a Private World Model.** The only durable competitive advantage in the age of AI will come from a system's ability to deeply understand an organization's unique operations, data, and "action space." This creates a private intelligence asset (a World Model), that compounds in value and cannot be replicated by competitors relying on public models.
2. **Continual Learning is the Manufacturing Process.** A World Model is not static; it is a living system. The defining bottleneck for building one is the inability of current systems to learn continuously from their operational environment. Continual learning is the manufacturing process that turns raw interaction data into refined, institutional knowledge.
3. **The Action Space is the Raw Material.** The unique data that fuels a World Model is the full history of actions, decisions, and outcomes within an organization. This "action space" is the most valuable, proprietary dataset an enterprise possesses. However, most companies lack the infrastructure to capture, structure, and learn from it.
This shift creates a strategic crisis: enterprises are investing billions to deploy agents, but the underlying stack is designed to consume intelligence, not create it. 
4. **Enterprise Spend is Now Committed.** Capital has decisively moved from experimentation to production. The surge past [$8.4B](https://menlovc.com/perspective/2025-mid-year-llm-market-update/) in enterprise API spend in H1 2025, on pace for $20B+ FY, is flowing primarily to production inference workloads. This signals operational dependence at scale.

This creates a strategic crisis: enterprises are investing billions to build compounding systems, but the underlying AI stack is architecturally static. It was designed to recall information, not acquire skill. As a result, the ROI on these massive investments is capped. The market now requires a new layer of infrastructure to solve this problem - moving from **next token prediction** to **next action prediction.**

![Shapes Sept 25 from Arc System Architecture.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Shapes_Sept_25_from_Arc_System_Architecture.png)

## **The Implication**

The direct consequence of this market shift is that **AI-native companies are hitting a wall.**

Having pushed prompt and context engineering to their limits, they are now managing overly complex, brittle agentic architectures with numerous points of failure. 

These workarounds are using strengths in one area (like a large context window) to compensate for a fundamental weakness (no long-term memory).

This leaves them stuck in a **Pareto trap**: choose between expensive, slow reasoning from a frontier model, or the operational overhead of fine-tuning smaller models. Neither approach solves the core problem of agent reliability and efficiency at scale.

![Arc Group 3.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Arc_Group_3.png)

**This leaves the enterprise market with a critical choice**: wait for the labs to build a generic solution, or build their own defensible, proprietary intelligence asset now.

Our position in the AI value chain is to provide the critical infrastructure that enables the latter.

Arc's **Learning Layer** is the missing piece of infrastructure that allows enterprises to build their own World Models instead of just renting intelligence. To fill this gap, we enable a fundamental architectural shift: **moving from reactive to proactive agentic reasoning.**

![world-model-architecture.jpg](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/world-model-architecture.jpg)

- **Top (The Current Paradigm):** An agent acts in the real world and receives feedback only after a potential failure. This loop is inefficient, expensive, and requires consistent evaluation.
- **Bottom (What We Build):** Instead of acting directly, the agent first queries its World Model. Inside this private simulation, it can imagine actions, simulate outcomes, and self-correct in a tight, low-cost internal loop. This allows the agent to iterate and refine its approach before committing an optimized action in the real world. **This is how we turn hindsight into foresight.**

Our architecture is the Learning Layer, the critical infrastructure that uncaps the ROI on existing AI investments and defines a necessary new category.

Here is how we see the AI value chain and Arc's unique position within it:

![Arc Group 4 (2).png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Arc_Group_4_(2).png)

## **Team**

Our founding team was purpose-built to design a tight loop between research and product needed to win in this market. 

### **Jarrod Barnes (Co-Founder): Ex-GM at Protocol Labs & NEAR Protocol, NYU Professor, Learning Design PhD**

Jarrod drives Arc’s technical vision, which is grounded in his Doctorate in Learning Design. His unique insight, that reliable AI must be built on the principles of how humans learn, is Arc’s core differentiator. He combines this with direct experience building and scaling both **complex distributed systems** and **developer ecosystems** as a General Manager at **Protocol Labs** and **NEAR Protocol**, providing the playbook for establishing Arc as the industry standard. His background also includes roles as an Assistant Professor at NYU, an investor at Emerson Collective, and a previous life as an NFL Front Office employee and College Football Coach at Ohio State and Clemson University. 

### **Gaby Haffner (Co-Founder): Cambridge MA, Ex-Farfetch enterprise platform lead, Former Founder, EY M&A, Ex-BAML**

Gaby drives Arc's commercial vision, translating our research into an enterprise business. As a former founder, she's built entire GTM functions from the ground up. Her background is translating infrastructure into a valuable enterprise product. At Farfetch, she led the team that productized its internal infrastructure into a white-labeled enterprise platform for Chanel, Thom Browne, and Calvin Klein. Her experience at Monitor Deloitte advising Fortune 500 C-suites and EY Investment Banking, ensures we can navigate complex enterprise deals. 

### **Michelangelo Naim (Founding Research Scientist)** MIT Postdoctoral Fellow, Weizmann Institute, Theoretical Neuroscience PhD

Michelangelo provides the deep scientific foundation for our vision. His research in theoretical and computational neuroscience at MIT and the Weizmann Institute  focused on the fundamental principles of intelligence. His PhD thesis, *"Episodic memory from first principles,"* directly influences Arc’s core mission: architecting systems where memory is not a database to be queried, but a skill to be acquired. He ensures our technology is built on a durable scientific moat. 

### **Aman Jaglan (Founding ML Engineer): NLP/Deep Learning researcher, Protiviti enterprise systems architect**

Aman leads engineering execution, translating our research into scalable, enterprise-grade software. His background in **NLP/Deep Learning research** and experience building robust systems for enterprise clients at **Protiviti** ensures our platform is both innovative and reliable. He architected production systems that handle complex enterprise workflows, exactly the type of infrastructure Arc needs to support Fortune 500 deployments.

## **Product**

Our open-source data engine provides the end-to-end manufacturing process for creating these World Models. **It's designed for minimal overhead, wrapping a customer's existing agent in just four lines of code to immediately begin capturing value from their live environment.** See 👉 https://github.com/Arc-Computer/ATLAS

![atlas-core-stack.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/atlas-core-stack.png)

### **Product Demo: The “J-Curve” of Learning**

[Atlas.sdk-high.mp4](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Atlas.sdk-high.mp4)

*This chart from a live demo shows a clear "J-curve" of learning for a supply chain agent. An initial dip in runtime efficiency is quickly followed by dramatic improvement as learning compounds. After just five scenarios, the agent achieves a **+25.8% runtime improvement** over its baseline, proving our system turns static agents into assets that get faster, cheaper, and more effective with every interaction.*

![atlas_demo_chart.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/atlas_demo_chart.png)

### Runtime Execution

![Shapes from Arc System Architecture (2).png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Shapes_from_Arc_System_Architecture_(2).png)

## **How Arc’s System Works**

### **1. Foundation: The Data Engine**

Atlas serves as a data engine to power the training of world models. We build this by capturing an agent's **early experience**. The system allows a customer's agent (the Student) to learn the basic cause-and-effect dynamics of its live environment by proposing actions and observing the resulting outcomes. This process of Implicit World Modeling turns the customer's operational environment into a hyper-efficient data generation engine, creating the rich, proprietary dataset of their unique "action space" that forms the World Model.

### **2. Reasoning Core: The Dual-Agent System**

The architectural center of Arc is its **Teacher-Student** model. This dual-agent system decouples strategic planning from tactical execution. 

Our Teacher model wraps the customer's existing agent (the Student) and guides it through live tasks. Every interaction (the Student's attempt, the Teacher's guidance, and the final outcome) is captured as a structured training asset. This turns the customer's live operational environment into a hyper-efficient data generation engine, creating a rich, proprietary dataset of their unique "action space". This is the raw material used to build and continuously refine the World Model.

> This process is proven to deliver a [6x performance uplift](https://drive.google.com/file/d/1M_271zXNeHnfiWZmLnOeYX9gyWMz2OjF/view?usp=sharing) and established a new [state-of-the-art on the τ²-bench benchmark](https://www.arc.computer/blog/Navigating-Dual-Control-Environments) for agent reliability.
> 

This methodology is the mechanism for **cross-domain skill transfer**, which has been proven to generalize from data-rich domains (like mathematics) to data-scarce enterprise environments.

### **3. Reward System: The Feedback Engine**

This entire learning loop is fueled by our proprietary Reward System, which recently achieved state-of-the-art on RewardBench V2. It translates complex, messy feedback from the live environment into a precise reward signal. This signal is critical for teaching the World Model to accurately simulate the consequences of different actions, forming the foundation of trustworthy, independent learning.

**Day 1 integration note:** The ArcOps-Cyber runtime now layers the Microsoft SecRL ArcOps evaluator atop our RIM telemetry. Student runs issue parameterised queries through the secured `secrl_sql` adapter (read-only MySQL on `localhost:3307`), and the SecRL reward is logged in-line with our J-curve diagnostics. This lets us claim apples-to-apples performance while keeping ATLAS’s pedagogical insight channels intact.

![reward-leaderboard (1).png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/reward-leaderboard_(1).png)

### **4. Learning Engine: Hybrid Offline and Online Training**

The engine that drives skill acquisition operates in two modes:

- **Offline (Reinforcement Learning):** In an offline environment, we use our open-source **ATLAS** framework to instill deep, foundational skills in Teacher agents. This process is benchmarked to deliver a [**+31% task completion rate** with a **97% non-degradation rate**](https://www.arc.computer/blog/introducing-atlas), ensuring reliability before deployment.
    
    ![offline.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/offline.png)
    
- **Online (Real-Time Prompt Adaptation):** In production, a **Genetic Prompt Evolution Algorithm (GEPA)** provides hyper-efficient, real-time adaptation. This online loop programmatically analyzes failures and evolves the Teacher's guidance, delivering a [**+165% performance gain in under two hours for less than $10 in inference costs.**](https://www.arc.computer/blog/supercharging-rl-with-online-optimization)

![online.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/online.png)

### **5. The Flywheel: Persistent Memory and Portable Skills**

This is Arc's key differentiator. Every learning trajectory is captured as **persistent memory**, which continuously enriches the World Model. This collected experience allows skills to become portable assets, transferable to new agents and new tasks. This entire mechanism for creating a compounding knowledge base is available out-of-the-box in our open-source framework.

> ***Note: Arc is model agnostic. Developers can use Arc’s pre-trained teachers, train their own, or use any closed/open source model.***
> 

<aside>
👉

**See [ATLAS](https://github.com/Arc-Computer/ATLAS) - our core, open-source engine for offline RL and online learning and our [HuggingFace](https://huggingface.co/Arc-Intelligence) for the repository of all of our models and datasets.** 

</aside>

[https://github.com/Arc-Computer/ATLAS](https://github.com/Arc-Computer/ATLAS)

## **GTM & Business Model**

Our Go-to-Market is a deliberate two-phase sequence designed to achieve product-market fit at high speed before scaling. We are using a targeted developer-first motion to earn the right to scalable, downstream partnerships.

![Shapes from Arc System Architecture (2).png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/Shapes_from_Arc_System_Architecture_(2)%201.png)

### **Phase 1 (0 → 1): Targeted ICP & OSS Distribution**

The singular goal of Phase 1 is to prove that Arc solves an acute, high-value problem for a specific, repeatable customer profile. We are surgically targeting the **High Dynamism x High Observability** quadrant, where the need is most urgent and the ROI is clear to prove (verifiable task domain). 

- **Our ICP & Wedge**: Our wedge is targeting **Series A-B+ companies whose core product depends on the reliability of multi-agent systems.** These teams have already pushed context engineering to its limits and are now struggling with brittle, complex workflows and unpredictable token costs.
- **We target CTO’s directly and ground our core value prop in efficiency (increased performance with decreased tokens) and speak directly to the acute need - this allows us to balance a motion of PLG and forward deployed by grounding a frontier capability with immediate and clear positioning.**
    - We compliment this with real world benchmarks that we’re currently collaborating on with Datadog , Auth0/Okta, (AI Security) and [Reply](https://www.reply.com/en) (AI CRM & Supply Chain)
    - Customers like **Treater**, who build multi-agent systems for CPG inventory management, validate this pain point, seeing immediate value in replacing brittle workflows with Arc's ability to learn and persist that skill across agents.
- **OSS Developer Adoption**: Our initial open-source push with ATLAS is designed to attract developers grappling with these exact use cases within our ICP. We will relentlessly focus on their experience to answer critical questions: Which integrations are mission-critical? What are the breakout use cases in this quadrant? What metrics define "success" for them? This provides the ground truth we need to build a compelling commercial product

Our definition of success here is a repeatable playbook for the top-right quadrant: a validated set of integrations, proven high-value use cases, and a crystal-clear value proposition that resonates with our target enterprise buyer.

### **Phase 2 (1 → 10): Converting Momentum into an Enterprise Platform**

The learnings from Phase 1 directly de-risk and inform our enterprise platform strategy, allowing us to expand into other high-value quadrants.

**Goal**: Convert our most engaged open-source users into paying customers on our managed enterprise platform. The sales cycle is dramatically shortened because the value has already been proven within their own organization

- **Platform Expansion**: The enterprise platform, powered by our **Reward Interpretation Model (RIM)**, is designed to capture the **Low Observability x High Dynamism** quadrant. For tasks like autonomous long-context drafting or aligning to subjective user preferences, the RIM's ability to translate messy feedback into dense reward signals is a core differentiator, unlocking a massive new set of enterprise workflows

**Partnerships & Monetization**

- **Distribution Partnerships:**
    - **Upstream:** To accelerate our learning with developers, we are building lightweight, framework-agnostic wrappers for the tools they already use (e.g., OpenAI’s Agent SDK, LangGraph). For developers within our ICP, this makes adoption frictionless, a simple import that immediately proves the value of our learning layer on their existing agents. This is our primary mechanism for the high-speed learning motion in Phase 1.
    - **Downstream:** Once we have a proven playbook, we will activate our downstream channels. We will partner with emerging neocloud and MLOps platforms, providing ATLAS as an embeddable engine (ie. a reinforcement fine-tuning endpoint) giving us scaled access to their enterprise customer base.
- **V1 Monetization Plan:**
    - **Base Platform Fee:** A recurring monthly fee for our managed cloud service. This covers the core infrastructure for organization-specific memory and the compute for offline **'Teacher' model training**.
    - **Usage-Based Pricing:** Active consumption metered by the volume of **online learning loop inference**. This ensures our revenue scales directly with the real-time value and performance improvements our customers experience.

This phased approach builds momentum through widespread developer adoption, then layers on a high-ACV commercial motion. 

## **Vision**

Arc is building infrastructure for intelligence that persists and compounds across time and systems. The transition from reactive, single-session agents to proactive agents with **persistent, compounding World Models** is a technological inevitability.

When intelligence can accumulate experience within its own World Model, everything changes:

- **Agents stop being tools and become colleagues** - they have histories, specializations, and track records stored within their World Model.
- **Experience becomes a measurable asset** - the verified capabilities of a World Model can be quantified.
- **Learning becomes an asset class** - accumulated knowledge within these models has transferable worth.

Our infrastructure is the foundation for this future. By enabling any organization to build, refine, and deploy its own World Models, we are creating a network where skills become portable assets. This establishes the foundation for a marketplace where enterprises can lease, trade, and deploy agents based on proven, compounding experience, turning intelligence itself into the next great asset class.

## **Ask**

We are raising a **$3M pre-seed round** to execute on a clear 18-month plan to build our foundational technology, prove its value in the market, and establish Arc as the standard for how agents learn.

### Use of Funds

- **55% Talent:** Hire 2 senior ML engineers (RL focus) and 1 ML/SWE generalist.
- **20% Compute:** Secure compute for continuous training, fine-tuning of our Reward Interpretation Model, and scaling RCL for our first 10 customers.
- **15% GTM & OpEx:** GTM, legal, operations, and core infrastructure.

*Note: Operating model available upon request.*

[Investor FAQ](https://www.notion.so/Investor-FAQ-272778a1210a80fcac08f3d6ae3a9e6b?pvs=21)

## Research Publications & Benchmark Performance

The fundamental question driving Arc is ***“If we were to build an agentic system that learns like a human, what would its architecture look like?”***

Current agentic architecture is built for knowledge recall, not for skill acquisition.

Our first principle is that true, adaptable intelligence is not a static property but an emergent one, born from the continuous loop of action, feedback, and adaptation. Humans don't learn from brute-force optimization alone; we learn through curriculum, mentorship, and the transfer of abstract skills, principles found in educational psychology.

This insight has guided a multi-stage research that validates every layer of our technical architecture.

**Thesis - Identifying the Core Problem:** 

We first articulated that the future of AI value is not in static knowledge, but in dynamic learning systems. We identified the "Outer Loop", the process by which a model learns from experience, as the most critical and underserved layer of the AI stack.

- [*The Era of the Outer Loop: Building Compounded Intelligence*](https://www.arc.computer/blog/The-Era-of-the-Outer-Loop)
- [*Why We Train Models: We Build Skill, Not Just Recall*](https://www.notion.so/Why-We-Train-Models-We-Build-Skill-Not-Just-Recall-25d778a1210a81d6a48df4ef18fac9f7?pvs=21)

**Methodology - A New Approach to Training:** 

To solve the learning problem, we developed a novel training paradigm: Reinforced Continual Learning (RCL). This framework introduces our core Teacher-Student architecture and our key innovation of cross-domain learning, where a "Teacher" model masters reasoning in one domain to guide a "Student" in another.

- [*Introducing Arc: Reinforced Continual Learning for an Agent-First World*](https://www.arc.computer/blog/Reinforced-Continual-Learning)

**Framework - Open Sourcing the Engine:** 

We then productized our RCL methodology into ATLAS (Adaptive Teaching and Learning Alignment System), our open-source framework. The ATLAS technical report details our unique two-pass "diagnostic probing" protocol, which allows a Teacher agent to adapt its guidance, delivering a validated **+15.7% gain in accuracy** and a **+31% increase in task completion** with a **97% non-degradation rate.**

- [*Technical Report: ATLAS: Adaptive Teaching and Learning Alignment System for RL*](https://github.com/Arc-Computer/ATLAS/blob/main/docs/ATLAS-Technical-Report.pdf)
- [*ATLAS Online: Supercharging RL Models with Hyper-Efficient Online Optimization*](https://www.arc.computer/blog/supercharging-rl-with-online-optimization)

**Validation - Proving Performance in Complex Environments:** 

Finally, we stress-tested our framework on the hardest public benchmarks. Our NYRL-accepted paper, *Bridging the Judgment Gap*, proves the efficacy of our approach in complex, dual-control environments where both an agent and a human can act. On these tasks, our system demonstrated a statistically significant **6x performance gain** over a baseline agent, establishing a new state-of-the-art for agent reliability.

- [*NYRL Accepted Paper: Bridging the Judgment Gap: Cross-Domain Reasoning Transfer*](https://drive.google.com/file/d/1M_271zXNeHnfiWZmLnOeYX9gyWMz2OjF/view?usp=sharing)
- [*Mind the Gap: Navigating Dual-Control Environments with RCL*](https://www.arc.computer/blog/Navigating-Dual-Control-Environments)

## **Benchmark Performance Highlights**

### **Offline Learning (ATLAS): Foundational Reliability & Efficiency**

Our ATLAS framework demonstrates consistent and significant gains in a controlled, offline setting. The teacher-led approach not only improves accuracy and task completion but does so with far greater resource efficiency and, crucially, without degrading existing skills.

![offline.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/offline%201.png)

*The ATLAS framework delivers a **+15.7% average accuracy gain** and ensures **100% task completion**, all while reducing token usage by **37.2%**. The **97% non-degradation rate** proves its reliability.*

### **Online Learning: Hyper-Efficient Adaptation**

Our online learning loop delivers dramatic performance improvements in a live environment with minimal time and cost. This proves our ability to rapidly adapt a pre-trained Teacher to a specific task.

![online.png](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/online%201.png)

*In a live environment, our online optimization layer achieved a **+164.9% performance improvement** in approximately **2 hours**, using less than **$10 in inference costs.** This demonstrates the hyper-efficiency of our hybrid learning architecture.*

### **Additional Benchmark Demo:**

We put our system to the test in a real environment  on [IT bench](https://github.com/itbench-hub/ITBench-Scenarios/tree/main/sre), which simulates complex, cascading failures in Kubernetes. **To date, frontier models fail on over 85% of these tasks.** 

> We deploy an autonomous SRE agent within a live Kubernetes cluster and show the full learning loop in action. Claude 4.5 Sonnet fails on its own, our Teacher World Model provides real-time corrective guidance to steer the agent to success, and our reward system quantifies the learning, proving the system gets measurably smarter from each interaction.
> 

[arc-final-demo.mp4](Arc%20Pre-Seed%20Investment%20Memo%20271778a1210a80b69da5d1954987f68a/arc-final-demo.mp4)

*This video shares the full loop of our teacher and student interaction, with real time rewards (via our reward system) and prompt optimization to adjust the models behavior in real time.*
