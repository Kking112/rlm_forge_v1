## Rules

- Your project **must** use OpenEnv (stable release 0.2.1) deployed on HF spaces

- You must show a minimal training script for your environment using Unsloth or HF TRL in Colab.

- You must upload a **one minute** demo video to YouTube talking about your submission.

## Hackathon Problem Statements**

Your project must address at least **one of the five** required problem statements.

- Some problem statements include **optional partner-sponsored sub-problem statements**, which are additional focus areas related to the main theme.

- Your project may align with **multiple partner sub-problem statements**, but you can only be **judged for a maximum of two**. Please **select up to two** when submitting.

- Projects that match these partner sub-problem statements are eligible for **extra partner prizes**, judged separately from the main track winners.

- Each partner sub-problem statement carries a prize of **$10,000 USD**.

**Statement 1: Multi-Agent Interactions**

Environments for this theme involve cooperation, competition, negotiation, and coalition formation. Learning from these environments will enable agents to model the beliefs and incentives of others in partially observable settings. This drives theory-of-mind reasoning and emergent strategic behavior.

- **Expected Outcome:** an environment that can be used to train multi-agent task handling in a LLM

- **Example Environments:** Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

- **Partner Sub-Themes:**

  - **Fleet AI:** Scalable Oversight: Environments that train oversight agents to monitor, analyze, and explain the behavior of other AI agents operating in complex, multi-agent settings.
  - **Halluminate:** Multi-Actor Environments: Build a realistic environment where an agent interacts with and manages multiple actors (agents) to discover and achieve the task

**Statement 2: (Super) Long-Horizon Planning & Instruction Following**

You will build environments that require deep, multi-step reasoning with sparse or delayed rewards. After using these environments, the goal is to enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes. The aim is to push beyond shallow next-token reasoning toward structured planning and durable internal representations. 

- **Expected Outcome:** an environment that can capture and improve LLM behaviour on challenging long horizon tasks that need long running sessions beyond context memory limits. 

- **Example Environments:** Research-planning simulators, large-scale codebase refactoring tasks, strategic resource management worlds, long-horizon logistics optimization, extremely complicated long-horizon instruction following (e.g., 300 instructions scattered around).

- **Partner Sub-Themes:**

  - **Mercor:** Make an environment with capped/uncapped rewards where frontier model rewards scale with token output.

  - **Scale AI:** Environments for long horizon workflows for non-code use cases within a business setting: focusing on either Sales, Project management, or HR & IT.

**Statement 3: World Modeling**

- **Statement 3.1: Professional Tasks:** Here you will develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts to arrive at the desired outcome. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows. The goal is to strengthen causal reasoning and persistent world models.

  - **Expected Outcome:** an environment capturing nuances of a defined partially observable world and improve LLM interaction with it

  - **Example Environments:** Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops (papers → code → experiments), economic simulations with feedback, tool-discovery benchmarks.

  - **Partner Sub-Theme:**

    - **Scaler AI Labs:** Multi-App RL Environment for Enterprise Workflows: Create RL environments to demonstrate complex workflows, business rule nuances etc in a large enterprise

- **Statement 3.2: Personalized Tasks:** Here we will develop an environment that offers real personalized task handling, imagine replying to personal messages or handling dinner conflicts due to work conflicts, replying to tough emails. Think any personal assistant tasks.

  - **Expected Outcome:** An environment that gives the model a realistic simulation of handling personal tasks, conflicts and managing them as delegations

  - **Example Environments:** Executive Assistant Meeting Planner, Dinner and drive planning, email and message replying, etc

  - **Partner Sub-Theme:**

    - **Patronus AI:** Consumer Workflows with Schema Drift: Multi-step consumer workflow environments where the underlying data schemas, API contracts, and t&cs/policies/rules change.

**Statement 4: Self-Improvement**

The focus here is to create environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own capability growth. The objective is recursive skill amplification.

- **Expected Outcome:** an environment for improving self-play of a LLM over a defined set of tasks

- **Example Environments:** Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

- **Partner Sub-Theme:**

  - **Snorkel AI:** Simulated Experts-in-the-Loop: Environment that simulates interactions with real subject-matter experts, with changing requirements / preferences.

**Statement 5: Wild Card - Impress Us!**

We do not want to limit your focus if your idea doesn’t fit the boxes above, we want and WILL reward out of box tasks, please be creative but remember to add submissions that meaningfully add value to LLM training on a certain task. 


**Judging Criteria**

- **Environment Innovation (40%) -** Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?
- **Storytelling (30%) -** Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?
- **Training Script Showing Improvement in Rewards (20%) -** Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)? 
- **Reward and Training Pipeline Setup (10%) -** Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent’s inference (how it acts in the environment)?

**Judging Process**

**|** Judging proceeds in two rounds:

- Hackers will be assigned groups of judges; \~3 minutes to pitch followed by 1-2 minutes of Q/A

- The top **six** teams in ranking will get to demo on stage to a panel of judges; \~3 minutes to pitch followed by 2-3 minutes for Q/A.

## **11. Prizes**

- **1st Place:** $15,000 USD Cash

- **2nd Place:** $9,000 USD Cash

- **3rd Place:** $6,000 USD Cash