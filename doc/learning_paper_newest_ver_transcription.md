# Learning Common Norms in Multi-Agent Systems

_Transcribed from PDF (automated text extraction)._

---

## Page 1

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
Learning Common Norms in Multi-Agent Systems
VEDIC SHARMA and PETER MARBACH
ACM Reference Format:
Vedic Sharma and Peter Marbach. 2026. Learning Common Norms in Multi-Agent Systems. 1, 1 (February 2026),
36 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 INTRODUCTION
How can autonomous agents, each pursuing individual goals with limited information, come
together to make collective decisions in complex environments? This question lies at the heart of
multi-agent systems research and remains an open, critical challenge. In situations where agents
must coordinate-whether by adopting shared strategies, interpreting observations collectively,
or establishing mutual understanding-the formation of common knowledge, often embodied in
shared norms, is essential. Without this foundation, agents risk misalignment, which can lead to
inefficiencies or even system failure. Solving this problem is paramount, not just for advancing
theory but for enabling high-stakes applications, from autonomous vehicle fleets to expansive,
distributed AI networks.
The significance of this challenge is especially evident when compared to human societies. In
social systems, common norms, cultural expectations, and institutional frameworks create shared
understanding, enabling individuals to navigate complex interactions. Over time, these norms
evolve to support coordinated action, even among people with diverse viewpoints and incomplete
information. Just as common norms stabilize social behavior and support collaborative decision-
making in society, common knowledge in multi-agent systems enables agents to work collectively
toward shared goals, despite differences in individual observations or objectives.
Yet, unlike human societies-where norms and shared knowledge develop organically, adapting to
the group’s diverse needs-current multi-agent systems lack established mechanisms for evolving
shared norms. This gap presents a fundamental obstacle to scalable, effective coordination in
multi-agent systems, especially in dynamic environments where agents must rapidly adapt to
change.
This paper addresses the critical challenge of enabling agents in multi-agent systems to au-
tonomously establish common norms. To achieve this, we explore reward structures as a mechanism
for the learning and emergence of shared norms. Our objective is to design a reward structure that
satisfies three essential properties:
(1) Each agent selects actions that maximize its own reward within the specified reward
structure.
(2) Collectively, agents converge to a common norm.
(3) The resulting common norm is (close to) socially optimal, maximizing (or nearly maximizing)
the overall social welfare of the system.
Authors’ address: Vedic Sharma; Peter Marbach.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2026/2-ART
https://doi.org/10.1145/nnnnnnn.nnnnnnn
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 2

50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
2 Vedic Sharma and Peter Marbach
The reward structure we consider is inspired by social group dynamics and consists of three
types of rewards that agents can choose to optimize:
(1) Personal Utility : An agent’s intrinsic rewards based on their preferences.
(2) Providing Utility to Others : Agents extrinsic rewards gained from benefiting other agents.
We refer to this reward as reputation.
(3) Providing Information : Agents extrinsic rewards gained from sharing information about
which norms lead to high reputation. We refer to this reward as status.
Each agent focuses on the reward type that provides the highest incentive. Depending on the
reward they prioritize, agents perform distinct roles within the group. While different agents may
take on different roles, we show that each reward type-and the corresponding role-is essential for
a common norm to emerge. Notably, the presence of all three reward types is necessary to support
the emergence of a shared norm.
Intuitively, if all agents optimize their reputation by choosing norms that provide the highest
utility to others, a common norm that maximizes social welfare should emerge. To assess their
reputation, agents interact with others to gauge how well their norms are received. However,
reputation learning faces two main challenges: first, interacting agents may not provide accurate
feedback due to limitations in knowledge or inherent biases. Second, agents would need to interact
with all other agents to obtain an accurate assessment of their reputation, which becomes infeasible
in large systems.
We show that these challenges can be overcome by incorporating personal utility and status as
intrinsic rewards within the reward structure. With a reward structure combining personal utility,
reputation, and status, the emergence of a common norm proceeds through the following stages:
(1) Personal Utility Optimization : Agents initially adopt norms that maximize their own personal
utility.
(2) Reputation Learning : Although agents cannot directly observe their own reputation, they
can evaluate the personal benefits gained from norms used by others. Through interactions,
agents exchange information about the estimated reputations of peers, learning which
norms are associated with higher reputations.
(3) Reputation Optimization: Agents adopt the norms of peers with high reputations as reference
points, aiming to increase their own reputation by emulating these influential norms. Agents
whose norms become widely adopted are termed influential agents .
(4) Status Optimization : Once recognized as influential, agents seek to reinforce their status by
increasing community engagement and adopting norms that further solidify their recogni-
tion.
(5) Emergence of a Common Norm : As members increasingly emulate influential agents, these
agents gain insights into the preferences of the group, enabling them to adopt norms that
maximize social welfare. These reciprocal interactions lead to the emergence of a socially
optimal common norm.
In Section 3, we provide a concrete example to illustrate this process. This progression resembles how
norms emerge when humans interact in a social group or community, making the process intuitive
and easily understood. This connection to norm formation in social groups and communities
suggests that such groups may indeed benefit from a reward structure similar to the one we
propose. Moreover, it implies that norms emerging through this structure are likely to be (close to)
socially optimal.
While the emergence of a common norm intuitively mirrors how humans interact in social
settings, the mathematical model and analysis required to formalize and prove convergence to a
socially optimal norm are more complex. This complexity stems from the multi-component reward
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 3

99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
Learning Common Norms in Multi-Agent Systems 3
structure, which includes personal utility, reputation, and status. Each component incentivizes
distinct behaviors, leading to a layered interdependence between agents’ actions and decisions
that would not arise with a single-component reward structure. Crucially, incorporating all three
components is necessary because agents cannot directly measure their own reputation, requiring a
combination of feedback mechanisms to drive socially beneficial outcomes.
In summary, the paper addresses a significant gap in the field of multi-agent systems: how
autonomous agents can effectively coordinate and establish common norms in decentralized
environments without the need for central authority or pre-programmed rules. This is particularly
challenging when agents are self-interested and have only limited, local information. The gap
lies in developing mechanisms that enable the emergence of socially optimal collective behaviors
from individual, decentralized decision-making, which is crucial for the reliability and efficiency of
complex distributed AI systems.
The most important contributions of this paper revolve around its novel three-component reward
structure (personal utility, reputation, and status) and the theoretical framework built upon it. By
demonstrating that this reward system, combined with a best-response dynamic, can lead to the
emergence of welfare-optimal common norms, the paper provides a foundational solution to the
coordination problem. Its potential impact is substantial, offering a blueprint for designing more
robust, adaptive, and scalable multi-agent systems where agents can learn to cooperate and achieve
collective goals autonomously, without explicit human intervention or centralized control. This
could revolutionize applications ranging from autonomous vehicle fleets and robotic swarms to
distributed resource management and large-scale AI networks.
More precisely, the key contributions of the paper are as follows:
•Novel Reward Structure for Norm Emergence : The paper introduces and formalizes a unique
three-component reward structure (personal utility, reputation, and status) that is crucial
for the emergence of common, socially optimal norms in decentralized multi-agent systems.
It argues that the presence and interplay of all three components are essential for agents to
converge on a shared norm, especially in overcoming challenges like inaccurate feedback
in reputation systems.
•Framework for Autonomous Coordination : It provides a foundational framework for how
autonomous agents, each with individual goals and limited information, can collectively
arrive at socially optimal decisions. This addresses a critical, open challenge in multi-agent
research by demonstrating a mechanism for establishing common knowledge and shared
norms without the need for centralized control or explicit communication beyond the
reward signals.
•Theoretical Performance Guarantees : The paper proposes to rigorously analyze the conver-
gence of agents to a common norm using a game-theoretic framework, specifically through
the concept of a Nash equilibrium. In addition, in addition it provides learning algorithms,
based on distributed stochastic gradient ascent, that enables agents to learn and optimize
their local rewards. We show that the resulting dynamic leads to the emergence of a common
norm that aims to maximize the overall social welfare.
•Winner-Takes-All Effect: The proposed reward system gives rise to a winner-takes-all effect,
in which agents with slightly higher reputation or status quickly attract the attention
and imitation of others. This positive feedback loop not only facilitates the emergence
of a common norm-by encouraging agents to align their behavior with that of the most
reputable or influential peers-but also significantly accelerates the convergence process,
enabling rapid coordination in distributed systems. Importantly, under our proposed reward
structure, the norm that emerges through this winner-takes-all dynamic is not arbitrary: we
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 4

148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
4 Vedic Sharma and Peter Marbach
show that it is (close to) socially optimal, maximizing or nearly maximizing overall social
welfare. In doing so, our model provides a principled mechanism by which self-interested
agents, through local interactions and social feedback, can autonomously achieve both
consensus and collective efficiency without the need for centralized control.
•Connection to Human Social Group Dynamics: More broadly, the overall dynamic of our
model - involving reputation accumulation, status-seeking behavior, the emergence of
opinion leadership, and the role of social feedback mechanisms akin to gossip - strikingly
mirrors patterns observed in human social groups. In real-world communities, agents are
conscious of their reputation and status and adapt their behavior to enhance both. Moreover,
gossip plays a central role, as individuals discuss and assess the actions and impact of others’
behavior. Our model and analysis provide a formal explanation of these mechanisms: we
show that a reward structure based on reputation and status, combined with gossip to
evaluate the standing of others, leads to a desirable outcome - namely, the emergence of
a common norm that maximizes social welfare. This connection opens the door to new
interdisciplinary research at the intersection of sociology and AI, and offers a promising
foundation for understanding and designing the interaction of social and artificial agent
systems.
In essence, the paper offers a robust theoretical and practical framework for solving the problem of
decentralized norm formation, which is paramount for the future development and deployment of
complex high-stakes applications of multi-agent systems, including autonomous vehicle fleets and
robotic swarms, and distributed AI networks.
In addition, the proposed progression of norm emergence, moving from individual utility to
reputation and then status, mirrors the evolution of norms in human societies. This connection
provides valuable insights into how complex collective behaviors can arise from simple, local
interactions, potentially bridging concepts between artificial intelligence and social science.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 5

197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
Learning Common Norms in Multi-Agent Systems 5
2 PROBLEM FORMULATION
In multi-agent systems (MAS), effective coordination among agents operating within a shared
environment is essential for achieving system-wide goals. Across real-world applications-from
collaborative robotics and autonomous vehicle fleets to distributed sensor networks and social
media platforms-agents must make decisions that balance optimizing their individual rewards
with contributing to collective outcomes. A central requirement for effective coordination is for
agents to establish a norm: a shared understanding of the environment and appropriate actions
within it. This shared knowledge allows agents to synchronize their actions and work toward
maximizing the overall performance of the system.
However, the challenge does not end with establishing a common norm. Once a norm is developed,
agents face the additional task of selecting the shared understanding that yields the highest collective
benefit. Given that agents may have heterogeneous objectives and rewards, this selection has a
direct impact on both individual and system-wide outcomes. Thus, the goal is to select a norm that
maximizes social welfare-defined as the total expected reward across all agents in the system.
2.1 Model Components
To formalize this problem, we introduce a model that captures the essential elements of agent
coordination in a shared environment. These components reflect the dynamics of how norms
emerge, how agents interact, and how they seek to optimize collective outcomes. The detailed
mathematical formulation is provided in Section 2.2.
•Common Environment: We consider a system comprising 𝑁 agents that operate within
a shared environment, characterized by a finite number of states. States can represent
different situations or conditions that can occur in the environment.
•Role of Agents: Agents can take on two distinct roles: actor, when performing actions
directed at others, andparticipant, when observing or responding to others’ actions. Different
agents may allocate different amounts of time to each role.
•Agents’ Behavior as Actors: In each state of the environment, agents decide on which
action to perform. The actions they choose characterize the agent’s behavior.
•Agents’ Perceived Benefit as Participants: Agents in the participant role perceive a
benefit from each observed action of another agent, which can be positive if the action is
advantageous or negative if it is harmful.
•Heterogeneous Set of Agents: Different agents in the participant role may perceive
different benefits from the same action in a given state.
•Agents’ Interactions: Different agents in the participant role may perceive different
benefits from the same action in a given state.
•Norm: If all agents, in their actor role, adopt the same behavior, this behavior constitutes
the common norm of the group. A norm can represent not only specific actions but also
rules or beliefs that agents adopt in each state.
•Maximizing Social Welfare: Ideally, agents identify and adopt a norm that maximizes
social welfare across the entire group.
2.2 Mathematical Model
We formally represent these model components as follows.
2.2.1 Shared Environment. We consider a set of agentsA= {1,2,...,𝑁 }, with 𝑁 ≥2, that operate
in a shared environment characterized by a finite set of states S. Each state 𝑠 ∈S occurs with
probability 𝑝(𝑠).
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 6

246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
6 Vedic Sharma and Peter Marbach
In each state 𝑠 ∈S, agents select an action from a finite set X𝑠 of 𝐾𝑠 possible actions. Note in
each state 𝑠, that all agents choose actions from the same set of possible actions.
2.2.2 Role of Agents. Let 𝜇𝑎,𝑖 be the rate with which agent takes on the role as an actor in the
community, and 𝜇𝑝,𝑖 the rate with which 𝑖 takes on the role as a participant.
2.2.3 Agents’ Behavior as Actors. The behavior of agent 𝑖 is defined by a mapping 𝜋𝑖 : S→X 𝑠,
where 𝜋𝑖(𝑠)denotes the action chosen by agent 𝑖 in state 𝑠. Let Π denote the set of all possible
behaviors (mappings from states to actions).
2.2.4 Perceived Benefit as Participant. Agents are heterogeneous: agent 𝑖 ∈A receives a benefit
𝑢𝑖(𝑠,𝑥)from action 𝑥 ∈X𝑠 in state 𝑠 ∈S. The benefit can differ across agents, states, and actions.
2.2.5 Interactions As Actors. Let 𝜇𝑎,𝑖 ≥0 denote the rate at which agent 𝑖 interacts within the
group. Define the vector
𝜇𝑎, = (𝜇𝑎,1,...,𝜇 𝑎,𝑁)
for all agents’ rates. The probability that agent𝑖is active within the group at a given time is modeled
by a function 𝜃  𝜇𝑎,𝑖
 : [0,∞)→[ 0,1].
2.2.6 Interactions As Participant. Let 𝜇𝑝,𝑖 ≥0 denote the rate at which agent 𝑖 interacts within the
group. Define the vector
𝜇𝑝, = (𝜇𝑝,1,...,𝜇 𝑝,𝑁)
for all agents’ rates. The probability that agent𝑖is active within the group at a given time is modeled
by a function 𝜃  𝜇𝑝,𝑖
 : [0,∞)→[ 0,1].
2.2.7 Norm. If all agents adopt the same behavior 𝜋, i.e.,
𝜋𝑖(𝑠)= 𝜋(𝑠), ∀𝑖 ∈A,𝑠 ∈S,
then 𝜋 is the (common) norm of the group.
Ideally, agents identify, and adopt, a norm𝜋∗that maximizes social welfare given by
𝜋∗= arg max
𝜋∈Π
∑︁
𝑖∈A
𝜃  𝜇𝑝,𝑖
𝑈𝑖(𝜋),
where
𝑈𝑖(𝜋)=
∑︁
𝑠∈S
𝑝(𝑠)𝑢𝑖(𝑠,𝜋 (𝑠)).
2.3 Research Questions
For this setup, we study the following key questions that address both the dynamics of norm
formation and its implications for system-wide outcomes:
(1) Norm Emergence: How, and under what conditions, does a common norm 𝜋 emerge in a
fully distributed manner, based solely on local interactions between agents?
(2) Welfare-Optimal Norm: Under what conditions does the emerging norm correspond to the
welfare-optimal norm 𝜋∗that maximizes social welfare?
(3) Mechanisms and Algorithms: Can we design distributed mechanisms or learning dynamics
that reliably guide agents toward welfare-optimal norms?
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 7

295
296
297
298
299
300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329
330
331
332
333
334
335
336
337
338
339
340
341
342
343
Learning Common Norms in Multi-Agent Systems 7
2.4 Discussion
Our formulation focuses on the study of fully distributed multi-agent systems, in which agents
interact locally and make decisions independently. Specifically, we are interested in settings where
each agent aims solely to maximize its own reward, without centralized coordination or access to
global information. The collective behavior that emerges from these local decisions determines
whether the system as a whole converges to a norm.
A central question we explore is whether such decentralized dynamics can lead not only to the
emergence of a common norm, but to the emergence of a norm that is welfare-optimal - that is, a
norm that maximizes the social welfare across all agents. This mirrors dynamics seen in market
economies, where individual buyers and sellers, each with their own preferences and objectives, act
to maximize their own utility (buyers) or profit (sellers). Despite the absence of central planning,
their interactions can produce market equilibria that are socially efficient, a phenomenon famously
described as the “invisible hand” of the market.
In our setting, we ask: can we design a reward or incentive structure under which agents, by
individually maximizing their own rewards in a fully distributed manner, collectively converge to a
norm that maximizes social welfare? We seek conditions under which the self-interested actions of
heterogeneous agents align with the collective objective of welfare maximization. Our work aims
to characterize these conditions, and to develop distributed algorithms that enable such alignment
without requiring centralized control.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 8

344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
8 Vedic Sharma and Peter Marbach
3 REWARD STRUCTURE AND OPTIMIZATION
To create a mechanism that aligns individual agent incentives with the overall performance of
the system-and leads to the emergence of a common norm that maximizes social welfare-we
introduce two key components:
(1) Reward Structure: We propose an incentive scheme designed to guide agents toward a
welfare-optimal norm. This scheme combines three core rewards:personal utility, reputation,
and status.
We begin by providing the intuition behind these three rewards through a concrete example,
and then present their formal definitions.
3.1 Illustrative Example
While the mechanism of using three distinct core rewards may at first appear complex, its underlying
intuition is simple and mirrors familiar reward systems in social groups. In the following, we
highlight this connection using a simple example that not only illustrates the reward system but
also the interactions and dynamics that lead to the emergence of a common norm. This example is
intended to illustrate the concepts rather than serve as a practical case study.
The example illustrates:
(1) the three types of rewards: personal utility, reputation, and status,
(2) how agents optimize these rewards through interactions with other agents,
(3) how reputation learning and social support play important roles in helping agents optimize
their rewards,
(4) how members continuously alternate between two roles: as actors, when they perform
social actions directed at others, and as participants, when they observe or receive actions
from others,
(5) and how this reward system and these interactions lead to the emergence of a common
norm.
We use this reward structure, interactions, and feedback mechanisms (reputation learning and
social support) in our analysis.
Consider a gardening group in which members cultivate vegetables and flowers using diverse
practices-such as planting techniques, watering schedules, and pest control methods. Members
constantly shift between the roles of actors and participants: in some moments, they perform
actions that others observe, and in other moments, they are participants who witness or receive
the actions of others. The three types of rewards, and how agents pursue them in this group, are
illustrated as follows:
(1) Personal Utility Optimization (Actor role): Alice, acting as an experimenter in the group, tries
different fertilizers for her plants. Through trial and error, she discovers that a standard
fertilizer works best for her. Here Alice is an actor, while the rest of the group may observe
her experimentation in their role as participants.
(2) Reputation Learning (Participant role): In conversations with Bob, Alice hears about Carol, a
gardener well-known for her thriving garden and eco-friendly practices. In this role, Alice
is primarily a participant, receiving information and updating her perception of Carol’s
reputation. Carol’s reputation itself reflects her past actions as an actor that were observed
and valued by many participants.
(3) Reputation Optimization (Actor →Participant feedback loop): Inspired by Carol’s standing,
Alice seeks her advice. Carol, now acting, recommends an organic fertilizer and eco-friendly
techniques. Alice adopts these practices, and as her garden flourishes, participants in the
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 9

393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439
440
441
Learning Common Norms in Multi-Agent Systems 9
group notice. Their recognition feeds back into Alice’s reputation, showing how actions in
the actor role become reputational signals to others in the participant role.
(4) Social Support (Participants acting back): As Carol’s reputation grows, more members ap-
proach her for guidance. They not only observe and adopt her behavior as participants
but also act in turn by praising, endorsing, and sharing her advice with others. This feed-
back-participants rewarding Carol-constitutes thesocial support that signals and amplifies
Carol’s status.
(5) Status Optimization (Actor reinforced by participants): Carol interprets the social support
she receives as evidence of her elevated status. In her actor role, she responds by becoming
even more attentive and generous with her expertise, which further strengthens the esteem
and deference she receives from the group in their participant roles. This iterative cycle
reinforces her central status.
(6) Emergence of a Common Norm (Mutual reinforcement of actor and participant roles): As more
members follow Carol’s techniques and provide feedback on what works best for them,
her methods diffuse widely. The interplay of actor contributions and participant feedback
gradually consolidates into a shared set of gardening practices, establishing a common
norm that benefits the entire group.
Through these interactions, members not only maximize their individual rewards but also, in
their alternating roles as actors and participants, contribute to the establishment of shared practices.
The combination of visible actions and audience feedback gradually aligns behaviors, leading to
the emergence of beneficial community norms.
In the following sections, we formalize this intuition and analytically demonstrate how this
reward structure, combined with interactions among agents, drives the emergence of common
norms that optimize social welfare.
3.2 Reward Structure
We define three core rewards that serve as an incentive mechanism for agents:
3.2.1 Personal Utility: The personal utility quantifies how beneficial a given behavior𝜋𝑖 is to agent
𝑖. Formally, it is defined as
𝑈𝑖(𝜋𝑖)=
∑︁
𝑠∈S
𝑝(𝑠)𝑢𝑖(𝑠,𝜋𝑖(𝑠)), (1)
where 𝑝(𝑠)denotes the probability of state 𝑠, and 𝑢𝑖(𝑠,𝜋𝑖(𝑠))is the utility agent 𝑖 derives from
action 𝜋𝑖(𝑠)in state 𝑠. When maximizing its personal utility, agent 𝑖 selects the behavior that is
most beneficial to itself.
3.2.2 Reputation: The reputation of an agent characterizes how beneficial the behavior of agent 𝑖
is to other agents. Formally, it is defined as
𝑅𝑖(𝜋𝑖)=
∑︁
𝑗∈𝐶\{𝑖}
𝜃  𝜇𝑎,𝑖
 𝜃  𝜇𝑝,𝑗
𝑈𝑗(𝜋𝑖). (2)
The terms 𝜃  𝜇𝑎,𝑖
 and 𝜃  𝜇𝑝,𝑗
 weight the reputational effect of agent 𝑖’s behavior by (i) how often
𝑖’s actions are visible as an actor, and (ii) how much attention and evaluative weight participant 𝑗
assigns to others’ actions. Thus, reputation depends not only on the objective benefits conferred
but also on the visibility of actors and the attentiveness of audiences.
The perceived reward (value) from this reputation is given by
𝛾𝑅𝑖(𝜋𝑖),
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 10

442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478
479
480
481
482
483
484
485
486
487
488
489
490
10 Vedic Sharma and Peter Marbach
where 𝛾 > 0, and 𝑈𝑗(𝜋𝑖)is the benefit that agent 𝑖’s behavior provides to agent 𝑗. The constant 𝛾
represents the weight that agent 𝑖 places on adopting behaviors that benefit others and thereby
increase its reputation.
3.2.3 Status and Social Support: The status of an agent 𝑘 is characterized by how many agents
follow 𝑘and adopt its behavior, as well as by the benefit these agents derive from doing so. Formally,
it is defined as
𝑆𝑘(𝜋𝑘)=
∑︁
𝑖∈F𝑘
𝜃  𝜇𝑝,𝑖
 𝑅𝑖(𝜋𝑘). (3)
where 𝑅𝑖(𝜋𝑘)is the reputation that agent 𝑖 obtains by adopting agent 𝑘’s behavior.
We interpret
𝜃  𝜇𝑝,𝑖
 𝑅𝑖(𝜋𝑘)
as the social support of agent𝑖to agent 𝑘. Here, “social support” refers to the symbolic endorsement
or recognition an agent gains when others adopt their behavior. Each follower 𝑖 ∈F𝑘 provides
social support to agent 𝑘 proportional to the reputation 𝑅𝑖(𝜋𝑘)that they receive from adopting the
behavior of agent 𝑘.
Agents express or communicate their social support through their adoption of agent𝑘’s behavior,
which serves as a public signal of approval and endorsement within the community. In order
to adopt this behavior, followers typically interact with agent 𝑘 to learn and observe it, during
which they provide positive feedback proportional to the reputation they gain from agent 𝑘’s
behavior. This positive feedback can take various forms, including verbal acknowledgment and
other supportive actions. The status thus represents the total reward that agent 𝑘 perceives from
the social support of its followers.
The perceived reward from status is given by
𝜅𝑆𝑘(𝜋𝑘),
where 𝜅 > 0 is the status parameter. The constant 𝜅 represents the weight that agents place on
having their behavior adopted by others.
3.3 Role Optimization
In the following, we assume that agents focus on maximizing the reward component (personal
utility, reputation, or status) that provides them with the highest payoff at any given time. This
leads to the following dynamic interaction: when the reputation of an agent 𝑘 increases, it attracts
more followers who adopt its behavior, thereby boosting both its status and further reinforcing
its reputation. This creates a positive feedback loop, akin to a winner-takes-all effect, where an
agent with even a slightly higher reputation than others may ultimately attract all other agents
as followers, achieving a reputation and status far greater than any other agent. Our analysis
formalizes this intuition.
3.4 Reward Optimization
3.4.1 Optimizing Personal Utility. To maximize its personal utility, agent𝑖 selects a behavior 𝜋𝑖
such that
𝜋𝑖 = arg max
𝜋∈Π
∑︁
𝑠∈S
𝑝(𝑠)𝑢𝑖(𝑠,𝜋 (𝑠)).
3.4.2 Reputation Learning and Optimization. The reputation of an agent characterizes how benefi-
cial the behavior of agent 𝑖 is to other agents. Formally, it is defined as
𝑅𝑖(𝜋𝑖)=
∑︁
𝑗∈𝐶\{𝑖}
𝜃  𝜇𝑎,𝑖
 𝜃  𝜇𝑝,𝑗
𝑈𝑗(𝜋𝑖). (4)
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 11

491
492
493
494
495
496
497
498
499
500
501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
533
534
535
536
537
538
539
Learning Common Norms in Multi-Agent Systems 11
Note that agent 𝑖 must interact with others and gather information to identify which agent 𝑘 has
the highest reputation in the group. We refer to this process as reputation learning .
To maximize its reputation, agent 𝑖identifies the agent 𝑘∗with the highest observed reputation:
𝑘∗= arg max
𝑘∈A
𝑅𝑘(𝜋𝑘),
and adopts the behavior of agent 𝑘∗, i.e.,
𝜋𝑖 = 𝜋𝑘∗.
3.4.3 Status Optimization. To maximize its status, agent 𝑘 selects a behavior 𝜋𝑘 such that
𝜋𝑘 = arg max
𝜋∈Π
∑︁
𝑖∈F𝑘
𝜃  𝜇𝑝,𝑖
 𝑅𝑖(𝜋𝑘)
Note that this is equivalent to choose a behavior 𝜋𝑘 that maximizes agent’s 𝑘 reputation, and
𝜋𝑘 = arg max
𝜋∈Π
𝑅𝑖(𝜋).
3.5 Optimizing Interaction Rates as Actor
Agent 𝑖optimizes its interaction rate as an actor as follows.
Let 𝜋𝑖 be the current behavior of agent 𝑖, and let
𝑅max = max
𝑖∈A
𝑅𝑖(𝜋𝑖)
be the highest reputation over all agents. Furthermore let 𝐻𝑖 be agent 𝑖’s maximum reward over
the differen roles given by
𝐻𝑖 = max{𝑈𝑖(𝜋𝑖),𝛾𝑅𝑖(𝜋𝑖),𝜅𝑆𝑖(𝜋𝑖)}.
Finally, let 𝑢0 be the reward an agent obtains from interactions outside the group.
Given a total interaction rate budget 𝑀 > 0, agent 𝑖 chooses its interaction rate 𝜇𝑖, and its
interaction rate 𝜆𝑖 outside the group, using the following optimization problem,
max
𝜆𝑖,𝜇𝑎,𝑖 ≥0
𝜃  𝜆𝑎,𝑖
𝑢0 +𝜃  𝜇𝑎,min +𝜇𝑎,𝑖
 𝐻𝑖
s.t. 𝜆 𝑎,𝑖 +𝜇𝑎,𝑖 ≤𝑀.
(5)
3.6 Discussion
In the formulation above, we assume that agents observe the benefits of others’ actions accurately
and update their evaluations without error. This assumption ensures that the definitions of reputa-
tion and status are consistent with realized payoffs and allows us to conduct a clean game-theoretic
analysis of equilibrium behavior. Introducing imperfect observation at this stage would complicate
the analysis by conflating incentive effects with noise in information processing.
Later, when we consider learning dynamics, we relax this assumption and allow agents to form
reputational assessments based on noisy or incomplete information. In that setting, agents must up-
date beliefs under uncertainty, and reputational signals may diverge from actual contributions. This
extension provides a more realistic model of distributed systems where perception and evaluation
are inherently imperfect.
To simplify the analysis and notation, we assume that agents have fixed interaction rates as
participants. These interaction rates could also be treated as decision variables, similar to the
optimization of interaction rates by an actor. Doing so does not change the fundamental results
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 12

540
541
542
543
544
545
546
547
548
549
550
551
552
553
554
555
556
557
558
559
560
561
562
563
564
565
566
567
568
569
570
571
572
573
574
575
576
577
578
579
580
581
582
583
584
585
586
587
588
12 Vedic Sharma and Peter Marbach
or insights presented in this paper, but it would require a more general definition of the welfare-
optimal norm and would complicate both the Nash equilibrium analysis in Section ?? and the
analysis of the learning algorithms in Sections ??-??.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 13

589
590
591
592
593
594
595
596
597
598
599
600
601
602
603
604
605
606
607
608
609
610
611
612
613
614
615
616
617
618
619
620
621
622
623
624
625
626
627
628
629
630
631
632
633
634
635
636
637
Learning Common Norms in Multi-Agent Systems 13
4 ANALYTICAL FRAMEWORK
Under the proposed reward structure, agents choose to optimize the reward component-personal
utility, reputation, or status-that provides the highest payoff. In this section, we provide the
framework that we use for the analysis. Specifically, we define the key concepts, notations, and
roles that describe agent decisions and interactions. We then outline two complementary analytical
approaches: a game-theoretic analysis assuming perfect information, and a learning-based analysis
for distributed, information-limited environments. Finally, we specify the fundamental assumptions
on which our analysis relies. Together, these elements allow us to characterize when and under
which conditions common norms emerge under our proposed incentive structure-and whether
such norms align with social welfare.
4.1 Assumptions
For our analysis-both the Nash equilibrium analysis and the analysis of the learning algorithms-we
make the following three assumptions.
Our first assumption requires that the personal utilities 𝑈𝑖(𝜋)of all agents 𝑖 ∈A are bounded.
Assumption 1. There exists a bound 𝐵𝑈 such that
|𝑈𝑖(𝜋)|≤ 𝐵𝑈, 𝑖 ∈A,𝜋 ∈Π.
Assuming that the personal utilities of all agents are bounded by a constant 𝐵is a standard and
mild assumption commonly adopted in game-theoretic and learning analyses. Bounded utilities
ensure that payoffs remain finite, which is essential for guaranteeing the mathematical well-
posedness of the model and for applying convergence results in learning algorithms. Moreover,
this assumption aligns well with practical scenarios where utilities represent measurable quantities
that naturally have upper and lower limits. While it may exclude some theoretical models with
unbounded payoffs, it does not significantly restrict the applicability of our results in the context
of norm emergence and strategic interactions among agents.
The next assumption concerns a given group Aof agents. We assume that for at least one agent,
the behavior 𝜋𝑃𝑈
𝑖 that maximizes personal utility also leads to a positive reputation.
Assumption 2. There exists an agent 𝑖 ∈A such that∑︁
𝑗∈𝐶\{𝑖}
𝑈𝑗(𝜋𝑃𝑈
𝑖 )> 0.
Assumption 2 is mild and reasonable, as it simply ensures that individual incentives and social
evaluations are not entirely misaligned for all agents. In most social or multi-agent systems, it is
natural to expect that some behaviors that maximize personal utility (e.g., cooperation or fairness)
also generate positive social feedback. This assumption rules out pathological cases where every
personally optimal action is purely antisocial. However, it may fail in adversarial environments
where personal gain comes only at others’ expense, or in highly fragmented groups where conflicting
standards prevent any action from receiving positive reputation. In such settings, norm emergence
driven by reputation would be inhibited.
4.2 Approach
Depending on which reward component an agent optimizes (that is, which setP(Φ), R(Φ), or S(Φ)
they belong to), the agent plays a different role in the group. As part of the analysis, we characterize
the dynamics of how agents switch roles, the resulting decisions of each agent, and whether this
process leads to the emergence of a common norm-and ideally to the emergence of a common
norm that maximizes social welfare. To do this, we proceed as follows.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 14

638
639
640
641
642
643
644
645
646
647
648
649
650
651
652
653
654
655
656
657
658
659
660
661
662
663
664
665
666
667
668
669
670
671
672
673
674
675
676
677
678
679
680
681
682
683
684
685
686
14 Vedic Sharma and Peter Marbach
We first analyze this process using a game-theoretic framework, where we assume that each
agent has perfect knowledge of the current decisions of all other agents, and uses this perfect
information to decide which reward component to optimize. For this setup, we characterize when
there exists a Nash equilibrium that leads to the emergence of a common norm, and when there
exists a Nash equilibrium that leads to the emergence of a common norm that maximizes social
welfare. This Nash equilibrium analysis provides fundamental insights into whether and under
which conditions a common norm emerges under our proposed reward structure.
The assumption that agents have perfect information, which we use for the Nash equilibrium
analysis, is unrealistic for practical settings. To address this, we next consider the case where
agents do not have a priori perfect information, but can acquire information through interaction
with other agents. This leads to learning algorithms in which agents iteratively learn (estimate)
the payoffs under the different reward components and optimize the component that currently
yields the highest payoff. We again characterize for these learning algorithms whether and under
which conditions a common norm emerges under our proposed reward structure. This setup yields
realistic algorithms suitable for practical situations where agents must make decisions and learn in
a fully distributed environment.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 15

687
688
689
690
691
692
693
694
695
696
697
698
699
700
701
702
703
704
705
706
707
708
709
710
711
712
713
714
715
716
717
718
719
720
721
722
723
724
725
726
727
728
729
730
731
732
733
734
735
Learning Common Norms in Multi-Agent Systems 15
5 NASH EQUILIBRIA, OPINION LEADERS, AND WELFARE-OPTIMAL NORMS
In this section we carry out the Nash equilibrium analysis where we study whether agents pursuing
individual goals within the proposed reward structure eventually adopt a common norm 𝜋∗that
maximizes social welfare. For this analysis we assume that agents have perfect information : each
agent knows the role, rate allocation, and behavior of every other agent in the system. We then
define a Nash equilibrium as follows:
5.1 Notation and Definitions
For this, we use the following notation and definitions.
5.1.1 Group Configuration: Agent Decisions and Followers.Each agent 𝑖 ∈A must decide on its
behavior 𝜋𝑖, its interaction rate 𝜇𝑖 within the group, and its interaction rate 𝜆𝑖 outside the group.
Let
Φ𝑖 = (𝜋𝑖,𝜇𝑎,𝑖,𝜆𝑎,𝑖,𝜇𝑝,𝑖,𝜆𝑝,𝑖)
denote the decision vector of agent 𝑖, and let
Φ = (Φ1,..., Φ𝑁)
denote the decision vector across all agents 𝑖 ∈A.
In addition, let
F = (F1,..., F𝑁)
be the vector indicating the set of followers of each agent.
Finally, let
Ω = (Φ,F)
denote the configuration of the group, including the decisions and the follower sets of all agents.
5.1.2 Behavior that Maximizes Personal Utility.For each agent 𝑖 ∈A, let 𝜋𝑃𝑈
𝑖 be the behavior 𝜋𝑃𝑈
𝑖
that maximizes its personal utility 𝑈𝑖(𝜋𝑖), given by
𝜋𝑃𝑈
𝑖 = arg max
𝜋𝑖 ∈Π
∑︁
𝑠∈S
𝑝(𝑠)𝑢𝑖(𝑠,𝜋 (𝑠)).
Note that 𝜋𝑃𝑈
𝑖 does not depend on the decisions of other agents, but only on the preferences of
agent 𝑖.
5.1.3 Agents with the Highest Reputation.Furthermore, given a the decision vectorΦ = (Φ1,..., Φ𝑁),
let A∗(Φ)be the set of agents with the highest observed reputation, i.e.,
𝑘∗= arg max
𝑖∈A
𝑅𝑖(𝜋𝑖), 𝑘 ∗∈A∗(Φ).
5.2 Agents’ Role
Using these definitions, agents choose which reward component to optimize as follows for a given
configuration Ω = (Φ,F).
5.2.1 Agents that Optimize Reputation. All agents 𝑖 ∉ A∗(Φ)decide whether to optimize their
personal utility or their reputation. They will choose to maximize their reputation by adopting the
behavior of an agent 𝑘∗∈A∗(Φ)if
𝑅𝑘∗(𝜋𝑘∗)> 𝐵𝑅
and
𝛾𝑅𝑖(𝜋𝑘∗)> 𝑈𝑖(𝜋𝑃𝑈
𝑖 ),
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 16

736
737
738
739
740
741
742
743
744
745
746
747
748
749
750
751
752
753
754
755
756
757
758
759
760
761
762
763
764
765
766
767
768
769
770
771
772
773
774
775
776
777
778
779
780
781
782
783
784
16 Vedic Sharma and Peter Marbach
where
𝑈𝑖(𝜋𝑃𝑈
𝑖 )=
∑︁
𝑠∈S
𝑝(𝑠)𝑢𝑖(𝑠,𝜋𝑃𝑈
𝑖 (𝑠)).
Let R(Ω)be the set of all agents 𝑖 ∉ A∗(Φ)for which this condition holds.
To optimize its reputation, an agent 𝑖 ∈R(Ω)follows an agent 𝑘∗ ∈A ∗(Φ)and adopts its
behavior. If A∗(Φ)contains more than one agent, agent 𝑖 selects an agent 𝑘∗uniformly at random
from A∗(Φ).
5.2.2 Agents that Optimize Status.All agents𝑘∗∈A∗(Φ)decide whether to optimize their personal
utility or their status. They will choose to maximize their status if their observed status exceeds the
payoff from personal utility maximization, i.e.,
𝜅𝑆𝑘∗(𝜋𝑘∗|F𝑘∗)> 𝑈𝑘∗(𝜋𝑃𝑈
𝑘∗ ).
Let S(Ω)be the set of all agents 𝑘∗∈A∗(Φ)for which this condition holds.
5.2.3 Agents that Optimize Personal Utility.All agents 𝑖 ∈A that do not optimize reputation (i.e.,
𝑖 ∉ R(Ω)) and do not optimize status (i.e., 𝑖 ∉ S(Ω)) optimize their personal utility. Let P(Ω)be the
corresponding set of agents.
5.3 Nash Equilibrium
Definition 5.1. A configuration Ω∗= (Φ∗,F∗)with
Φ∗= (Φ∗
1,..., Φ∗
𝑁) and Φ∗
𝑖 = (𝜋∗
𝑖,𝜇∗
𝑖,𝜆∗
𝑖),
and
F∗= (F∗
1,..., F∗
𝑁),
is a Nash equilibrium if the following holds:
(1) For all agents 𝑖 ∈P(Ω∗)we have
𝜋∗
𝑖 = 𝜋𝑃𝑈𝑖
and
(𝜇𝑖,𝜆𝑖)= arg max
𝜆,𝜇

𝜃 (𝜆)𝑢0 +𝜃 (𝜇)𝜋𝑃𝑈𝑖

.
(2) For all agents 𝑖 ∈R(Ω∗), there exists 𝑘∗∈A∗(Φ∗)such that
𝑖 ∈F∗
𝑘∗ and 𝜋∗
𝑖 = 𝜋∗
𝑘∗,
and
(𝜆∗
𝑖,𝜇∗
𝑖)= arg max
𝜆,𝜇

𝜃 (𝜆)𝑢0 +𝜃 (𝜇)ˆ𝐽𝑘∗(𝜋∗
𝑘∗,𝜇∗
𝑘∗)

.
(3) For all agents 𝑖 ∈S(Ω∗)we have
Φ∗
𝑖 = arg max
Φ

𝜃 (𝜆)𝑢0 +𝜃 (𝜇)ˆ𝐽𝑖(𝜋,𝜇 |F∗
𝑖))

.
By definition, under a Nash equilibrium Ω∗, no agent has an incentive to unilaterally change
its decision-that is, its behavior, rate allocation or role. Consequently, a stable configuration is
reached.
In the remainder of this section, we study the conditions under which a Nash equilibrium Ω∗
with a common norm
𝜋∗
𝑖 = 𝜋∗, ∀𝑖 ∈A,
emerges, and whether this common norm maximizes social welfare.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 17

785
786
787
788
789
790
791
792
793
794
795
796
797
798
799
800
801
802
803
804
805
806
807
808
809
810
811
812
813
814
815
816
817
818
819
820
821
822
823
824
825
826
827
828
829
830
831
832
833
Learning Common Norms in Multi-Agent Systems 17
5.4 Suffcient Condition for The Emergence of a Common Norm 𝜋
We first provide a sufficient condition that guarantees the existence of a Nash equilibrium with a
common norm. We have the following result.
Lemma 1. Given a Nash equilibrium Φ∗, if there exists an agent 𝑘∗∈A who is followed by all other
agents, i.e.,
F∗
𝑘∗ = A\{𝑘∗},
then 𝜋∗
𝑘∗ is adopted by all agents in the group and becomes the common norm. That is,
𝜋∗
𝑖 = 𝜋∗
𝑘∗, ∀𝑖 ∈A.
The result of Lemma 1 follows directly from the fact that, by definition, all agents 𝑖 ∈F𝑘∗ adopt
the behavior of agent 𝑘∗. We omit the detailed derivation for brevity.
We introduce the following convention. For a Nash equilibrium Ω∗satisfying the condition of
Lemma 1, we refer to the corresponding agent𝑘∗as an opinion leader (or influencer), reflecting their
role as a central figure whose behavior determines the common norm of the group. Furthermore,
we refer to a Nash equilibrium Ω∗as described in Lemma 1 as a Nash equilibrium with opinion
leader 𝑘∗.
For a Nash equilibrium with opinion leader 𝑘∗, we introduce a slightly weaker notion of a Nash
equilibrium with a common norm that maximizes social welfare.
Definition 5.2. Let Ω∗be a Nash equilibrium with opinion leader 𝑘∗and common norm 𝜋∗= 𝜋∗
𝑘∗.
We say that the common norm 𝜋∗maximizes social welfare if
𝜋∗= arg max
𝜋∈Π
∑︁
𝑖∈A\{𝑘∗}
𝜃  𝜇𝑝,𝑖
𝑈𝑖(𝜋).
This definition states that we consider a Nash equilibrium with opinion leader 𝑘∗to have a
common norm that maximizes social welfare if that norm maximizes the total utility of the followers
of the opinion leader.
5.5 Existence of a Nash Equilibrium with Common Norm 𝜋∗
We then obtain the following result on the existence of a Nash equilibrium with a common norm
𝜋∗.
Proposition 1. Let Assumption 2 hold. Given 𝑁, 𝑀, and 𝜅 = 0, there exists a 𝛾0 > 0 such that for
all 𝛾 > 𝛾0, a Nash equilibrium with an opinion leader 𝑘∗and a common norm 𝜋∗exists.
We provide a proof of Proposition 1 in Appendix ??. The key idea behind the proof is that a
sufficiently large 𝛾 amplifies the reward for reputation to the point that agents achieve a higher
overall reward by maximizing their reputation rather than their personal utility. As a result, all
agents become followers and adopt the norm of the agent 𝑘∗with the highest reputation, who
thereby becomes the opinion leader.
Having established conditions for the existence of a Nash equilibrium with a common norm,
we next provide conditions for the existence of a Nash equilibrium with a common norm 𝜋∗that
maximizes social welfare.
Proposition 2. Let Assumption 2 hold. Given 𝑁 and 𝑀, there exist 𝛾0 > 0 and 𝜅0 > 0 such that
for all 𝛾 > 𝛾0 and 𝜅 > 𝜅0, there exists a Nash equilibrium with an opinion leader 𝑘∗and a common
norm 𝜋∗that maximizes social welfare as given in Definition 5.2.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 18

834
835
836
837
838
839
840
841
842
843
844
845
846
847
848
849
850
851
852
853
854
855
856
857
858
859
860
861
862
863
864
865
866
867
868
869
870
871
872
873
874
875
876
877
878
879
880
881
882
18 Vedic Sharma and Peter Marbach
We provide a proof of Proposition 2 in Appendix ??.
Proposition 2 extends Proposition 1 to the case of a welfare-optimal norm, introducing an
additional condition on the parameter 𝜅. The key idea behind the proof is that if the incentive
parameter 𝜅for optimizing status is sufficiently large, agent 𝑘∗will choose to optimize its status, as
this provides a higher reward than maximizing personal utility. Furthermore, if 𝛾 is sufficiently
large, all other agents will prefer to maximize their reputation by following agent 𝑘∗rather than
maximizing their own personal utility. As a result, there exists a Nash equilibrium with a single
opinion leader 𝑘∗as described in Lemma 1, and this opinion leader optimizes its status, leading to a
common norm that maximizes social welfare.
5.6 Convergence to a Nash Equilibrium with the Optimal Common Norm 𝜋∗
The results we have obtained provide conditions under which a Nash equilibrium with a common
norm that maximizes social welfare exists. But how likely is it that this norm will actually emerge
as agents dynamically adapt their behavior, role, and rate allocation based on the actions of other
agents? Here, we analyze this question in the context of the best-response dynamic .
The best-response dynamic consists of collective strategies that are improved through iterative
steps, where each agent updates its strategy based on the strategies of the other agents in the
previous iteration. Essentially, the best-response dynamic provides a natural mechanism by which
agents choose strategies that maximize their individual benefit at each iteration.
Formally, the best-response dynamic generates a sequence of decision vectors,
Ω1,Ω2,Ω3,...
where, given an initial decision vectorΩ1, each subsequent configruationΩ𝑛 for 𝑛 ≥2 is obtained as
follows: each agent𝑖 ∈𝐶updates its decision by maximizing its reward (as outlined in Sections 3.4.1-
5.2), using the configuration Ω𝑛−1 from the previous iteration as input.
We make the following assumption on the initial configuration Ω1 for our analysis.
Assumption 3. For the intitial configuration Ω1 = (Φ1,F1)with
Φ1 = (Φ1,1,..., Φ1,𝑁) and Φ1,𝑖 = (𝜋1,𝑖,𝜇1,𝑖,𝜆1,𝑖),
and
F1 = (F1,1,..., F1,𝑁),
we have that
𝜋1,𝑖 = 𝜋𝑃𝑈
𝑖
and
(𝜇∗
1,𝑖,𝜆∗
1,𝑖)= arg max
𝜆,𝜇

𝜃 (𝜆)𝑢0 +𝜃 (𝜇)𝜋𝑃𝑈𝑖

.
Furthermore, we have that
F1,𝑖 = ∅.
The assumption on the initial configuration states that, at the outset, all agents optimize their
personal utility. This is both a mild and practical assumption. In the early stages of interaction,
when agents have no information about the behavior, preferences, or strategies of others, it is
reasonable for them to act in a way that maximizes their own utility. Such behavior reflects the
natural starting point for decision-making in the absence of social knowledge or coordination, and
provides a neutral baseline from which patterns of interaction, learning, and adaptation can emerge
over time. Algorithm 1 describes the resulting best-response dynamic.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 19

883
884
885
886
887
888
889
890
891
892
893
894
895
896
897
898
899
900
901
902
903
904
905
906
907
908
909
910
911
912
913
914
915
916
917
918
919
920
921
922
923
924
925
926
927
928
929
930
931
Learning Common Norms in Multi-Agent Systems 19
ALGORITHM 1: Best-Response Dynamic
Data: 𝑁,𝑀,𝛾,𝛿 , and 𝑢𝑖(𝑠,𝑥)for all states 𝑠 ∈S and actions 𝑥 ∈X𝑠 in state 𝑠
Result: A sequence of allocations Φ1,Φ2,... , which converge to a Nash equilibrium Φ∗with the optimal
common norm 𝜋∗
𝑛= 1 ;
foreach 𝑖 ∈𝐶 do
(𝜋𝑖,𝜇𝑖,𝑖)← optimal strategy given agent 𝑖is self-interested ;
Φ1,𝑖 ←(𝜋𝑖,𝜇𝑖,𝑖);
end
Φ1 ←(Φ1,𝑖)𝑖∈𝐶;
while True do
𝑛= 𝑛+1 ;
foreach 𝑖 ∈𝐶 do
// Computations based on the allocations (𝜋𝑗,𝜇𝑗,ℎ𝑗) for all agents 𝑗 ≠ 𝑖
(𝑈,𝑅,𝑆 )← optimal reward based on each role agent 𝑖can be;
role𝑖′←arg max(𝑈,𝑅,𝑆 );
(𝜋′
𝑖,𝜇′
𝑖,role𝑖′)← optimal strategy for agent 𝑖given it has the role role𝑖′;
Φ𝑛,𝑖 ←(𝜋′
𝑖,𝜇′
𝑖,role𝑖′);
end
Φ𝑛 ←(Φ𝑛,𝑖)𝑖∈𝐶;
(𝜋𝑖,𝜇𝑖,role𝑖)←( 𝜋′
𝑖,𝜇′
𝑖,role𝑖′);
end
We say that the best-response dynamic converges to the configuration Ω∗, if we have that
lim
𝑛→∞
Ω𝑛 = Ω∗.
We then have the following result.
Proposition 3. Let Assumption 2 and Assumption 3 hold. Given 𝑁 and 𝑀, there exist 𝛾0 > 0
and 𝜅0 > 0 such that for all 𝛾 > 𝛾0 and 𝜅 > 𝜅0, the best-response dynamic converges to a Nash
equilibrium with an opinion leader 𝑘∗and a common norm 𝜋∗that maximizes social welfare as given
by Definition 5.2.
We provide a proof of Proposition 3 in Appendix ??.
Proposition 3 states that for sufficiently large 𝛾 and 𝜅, the best-response dynamic converges
to a Nash equilibrium with a common norm. While the conditions for convergence to a Nash
equilibrium with an optimal common norm are the same as the conditions for existence provided
in Proposition 2, the convergence may require a higher thresholds 𝛾0 and 𝜅0 for 𝛾 and 𝜅.
5.7 Discussion
The Nash equilibrium analysis offers novel insights to the literature on norm emergence and
welfare-optimal consensus in multi-agent systems. While prior work on social learning, network
games, and evolutionary models of norms (e.g., [? ], [? ], [? ]) has examined conditions for consensus
or convergence to stable behaviors, these models typically do not provide explicit conditions under
which the emergent norms are welfare-optimal. In contrast, we derive precise Nash equilibrium
conditions linking group size, social feedback strength, and incentive parameters (reputation and
status) to the emergence of shared norms that maximize collective welfare. Our procedural model
uniquely integrates dynamic social feedback processes (such as imitation and evaluation) with
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 20

932
933
934
935
936
937
938
939
940
941
942
943
944
945
946
947
948
949
950
951
952
953
954
955
956
957
958
959
960
961
962
963
964
965
966
967
968
969
970
971
972
973
974
975
976
977
978
979
980
20 Vedic Sharma and Peter Marbach
formal equilibrium analysis, bridging a gap between evolutionary and game-theoretic approaches.
Moreover, by identifying the star network structure as the ideal configuration for promoting
welfare-optimal norm convergence, our results formalize and extend longstanding intuitions from
social theory (e.g., on charismatic authority and focal agents) in a rigorous equilibrium framework.
To our knowledge, no prior work combines these elements to provide explicit welfare conditions
for norm emergence in this way. Concretely, our main contributions are as follows:
(1) Existence of common-norm equilibria under broad conditions: We rigorously char-
acterize conditions (e.g., sufficiently large populations, adequate budget rates, and social
preference parameters) under which Nash equilibria with common norms exist. We show
that these conditions are not restrictive, highlighting that norm convergence is not a fragile
or rare phenomenon.
(2) Emergence of welfare-optimal norms: Our analysis shows that, under the identified
conditions, not only do agents converge to a shared norm, but the norm maximizes social
welfare. This shows that individual strategic behavior can yield socially desirable outcomes,
addressing a fundamental tension in game-theoretic models of norm formation.
(3) Critical role of social preferences and cooperation: The parameters 𝜅and 𝛾, capturing
the value agents place on information sharing and on norms that benefit others, are shown to
be decisive for achieving both norm convergence and welfare maximization. This highlights
the importance of fostering cooperative incentives in the design of multi-agent systems.
(4) Role of group size 𝑁: The analysis provides the interesting and somewhat surprising
result that the larger the group size, the easier it becomes for a common norm-and one
that maximizes social welfare-to emerge.
(5) Best-response dynamics and decentralized convergence: Our results show that a
simple decentralized procedure-the best-response dynamic-leads to convergence to Nash
equilibria with welfare-optimal norms. This provides a constructive and feasible pathway
for norm emergence in realistic systems where agents iteratively adjust their strategies
based on local observations.
How useful are these insights? Do they allow us to make progress on the fundamental ques-
tion: How can decentralized societies or systems achieve stable, welfare-maximizing norms without
external enforcement or central coordination? Taken together, the findings indeed offer a novel and
significant contribution to this question. They show that endogenous dynamics-driven solely
by local incentives, imitation, and reputational feedback-can produce both equilibrium stability
and social optimality. Local strategic behavior, far from leading to disorder or inefficiency, can
generate shared norms that align individual incentives with collective welfare. This advances
our understanding of how normative behavior can arise organically in decentralized systems. It
demonstrates that robust and socially desirable outcomes can emerge naturally from the interplay
of individual decision-making and social feedback, without reliance on exogenous authorities,
designed rules, or explicit coordination mechanisms. These insights open promising avenues for
designing multi-agent systems, distributed algorithms, and institutional frameworks that harness
endogenous dynamics to achieve desirable outcomes in complex, real-world environments.
In the following section, we build on these results to derive a fully distributed learning algorithm
that does not require the assumption of perfect information made in our Nash equilibrium analysis.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 21

981
982
983
984
985
986
987
988
989
990
991
992
993
994
995
996
997
998
999
1000
1001
1002
1003
1004
1005
1006
1007
1008
1009
1010
1011
1012
1013
1014
1015
1016
1017
1018
1019
1020
1021
1022
1023
1024
1025
1026
1027
1028
1029
Learning Common Norms in Multi-Agent Systems 21
6 LEARNING ALGORITHMS FOR PERSONAL UTILITY, REPUTATION, AND STATUS
In the previous section, we analyzed how agents can achieve socially beneficial coordination by
optimizing one of three reward types: personal utility, reputation, or status. Each of these rewards
corresponds to a distinct role that agents can take within the system-self-interested individual,
follower, or opinion leader.
In this section, we study how agents can learn to optimize each reward type in a decentralized
setting, based on limited local information and interactions with other agents. We focus separately
on the learning dynamics for each role: first, how agents can learn to maximize their own personal
utility; second, how they can learn to identify and imitate peers with high reputation; and third,
how influential agents can learn to optimize their status by responding to social feedback.
Importantly, in this section, we assume that an agent’s role is fixed-that is, each agent is
committed to optimizing one of the three reward types. In the next section, we extend this analysis
to how agents decide which reward to optimize.
6.1 Definitions and Notation
In this subsection we introduce additional definitions and notation that we define the learning
algorithms that we consider.
6.1.1 Behavior Representation. Each agent 𝑖maintains a parameter vector 𝑤𝑖 ∈R𝑑, which defines
a probabilistic behavior 𝜋(𝑤𝑖; 𝑠,𝑥), representing the probability of choosing action 𝑥 ∈𝑋𝑠 in state 𝑠.
These probabilities are computed using a softmax over a scoring function 𝜙(𝑤𝑖;𝑠,𝑥):
𝜋(𝑤𝑖;𝑠,𝑥)= exp(𝜙(𝑤𝑖;𝑠,𝑥))Í
𝑥′∈𝑋𝑠 exp(𝜙(𝑤𝑖;𝑠,𝑥′)). (6)
The function 𝜙 could be linear or nonlinear; we remain agnostic to its specific form in this
analysis. We do however assume that the behavior representation 𝜙(𝑤;𝑠,𝑥)has the following
property.
Assumption 4. The behavior representation 𝜙(𝑤;𝑠,𝑥)is differentiable in the weights 𝑤, and its
derivative ∇𝑤𝜙(𝑤;𝑠,𝑥)is Lipschitz continuous and bounded.
6.1.2 Time-dependent Variables. In the following we consider online algorithms, where agents
at each time step iteratively update the weights 𝑤𝑖 of their policy representation, as well as their
interaction rate allocations, and their roles. For this, we use the following notation.
Let 𝑠(𝑡)be the state at time 𝑡, let 𝑤𝑖(𝑡)the weights of agent 𝑖at time 𝑡, and let 𝜇𝑖(𝑡)and 𝜇𝑖(𝑡)be
the interactions rates at time 𝑡.
Furthermore, let P(𝑡), R(𝑡), and S(𝑡), indicate which agents optimize their personal utility,
reputation, and status, respectively, at time 𝑡.
Also, let 𝑤𝑝𝑢
𝑖 (𝑡), 𝑤𝑟
𝑖 (𝑡), and 𝑤𝑠
𝑖 (𝑡), be the weights that agent 𝑖 for its behavior in the different
roles. Agent 𝑖 initializes these weights at time 𝑡 = 0, and then uses the learning algorithms that we
introduce in this section to update its weights for 𝑡 ≥1.
Finally, let 𝜇𝑎,𝑖(𝑡)and 𝜇𝑝,𝑖, be the interaction rates of agent 𝑖 within the community as an actor
(i.e. choosing and taking an action) and as a participant (i.e. observing actions of other agents), as
function of time 𝑡. Here, we assume that the interaction rates of agents as a participant are fixed
(do not change over time) and given by 𝜇𝑝,𝑖, 𝑖 ∈A, but agents update their interaction rates as an
actor 𝜇𝑎,𝑖(𝑡)over time as we describe below.
Using these rates, we define the probability that an agent is active at time 𝑡 as an actor, or as
a participant, using the function 𝜃, we for concreteness we assume that this function is given as
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 22

1030
1031
1032
1033
1034
1035
1036
1037
1038
1039
1040
1041
1042
1043
1044
1045
1046
1047
1048
1049
1050
1051
1052
1053
1054
1055
1056
1057
1058
1059
1060
1061
1062
1063
1064
1065
1066
1067
1068
1069
1070
1071
1072
1073
1074
1075
1076
1077
1078
22 Vedic Sharma and Peter Marbach
follows:
𝜃(𝜇)= 1 −exp −𝜇, 𝜇 ≥0.
In this section, we focus on how agents update their weights 𝑤𝑖(𝑡)and their interaction rates as
an actor 𝜇𝑎,𝑖(𝑡), in the next section with focus on how agents update their roles and the resulting
dynamic of the sets P(𝑡), R(𝑡), and S(𝑡).
Importantly, agent 𝑖 updates its weight 𝑤𝑖(𝑡)only if agent 𝑖 is active in the group at time 𝑡.
6.2 Set of Active Agents A𝑎(𝑡)and A𝑝(𝑡)
For each time step 𝑡 ≥1, we define the two sets A𝑎(𝑡)and A𝑝(𝑡).
A𝑎(𝑡)is the set of agents that are active as an actor at time 𝑡. Eeach agent 𝑖 is included into this
set with probability 𝜃(𝜇𝑎,𝑖(𝑡))independentaly from all other agents and choices, where
𝜃(𝜇)= 1 −exp −𝜇, 𝜇 ≥0.
A𝑝(𝑡)is the set of agents that are active as participants at time 𝑡. Eeach agent 𝑖is included into
this set with probability 𝜃(𝜇𝑝,𝑖)independentaly from all other agents and choices, where
𝜃(𝜇)= 1 −exp −𝜇, 𝜇 ≥0.
Recall that the the rates 𝜇𝑝,𝑖 are fixed for all agents, and do not change over time.
6.3 Optimizing Personal Utility
We begin by defining how an agent 𝑖 ∈A𝑎(𝑡)that is active in the group at time 𝑡 updates its
weights 𝑤𝑝𝑢
𝑖 (𝑡)to maximize its personal utility.
6.3.1 Objective Function. In this case, the goal of agent 𝑖is to learn parameters 𝑤𝑝𝑢
𝑖 that maximize
its expected utility under the policy 𝜋(𝑤𝑝𝑢
𝑖 ; ·,·):
𝐽𝑝𝑢(𝑤𝑖)=
∑︁
𝑠∈𝑆
𝑝(𝑠)
∑︁
𝑥∈𝑋𝑠
𝜋(𝑤𝑖;𝑠,𝑥)𝑢𝑖(𝑠,𝑥), (7)
where 𝑢𝑖(𝑠,𝑥)is the utility received by agent 𝑖 for taking action 𝑥 in state 𝑠.
6.3.2 Gradient Estimation and Update. To optimize 𝐽(𝑤𝑖), the agent 𝑖is using a standard stochastic
gradient method. At time 𝑡, the agent 𝑖first choose an action 𝑥𝑖(𝑡)where the action 𝑥𝑖(𝑡)randomly
according to the random policy 𝜋(𝑤𝑖(𝑡); 𝑠(𝑡),𝑥)for the current weight vector 𝑤𝑖(𝑡). Next, agent 𝑖
updates its weight vector by setting
𝑤𝑝𝑢
𝑖 (𝑡+1)= 𝑤𝑝𝑢
𝑖 (𝑡)+𝛼(𝑡)·𝑢𝑖(𝑠(𝑡),𝑥𝑖(𝑡))·∇ 𝑤 log 𝜋(𝑤𝑝𝑢
𝑖 (𝑡);𝑠(𝑡),𝑥𝑖(𝑡)), (8)
where 𝛼(𝑡)is a stepsize parameter, and 𝑢𝑖(𝑠(𝑡),𝑥𝑖(𝑡))is the personal utility of agent 𝑖 in state 𝑠(𝑡)
and action 𝑥𝑖(𝑡).
This weight update increases the likelihood of actions that yield higher utility and reduces the
likelihood of less effective actions.
6.4 Reputation Learning and Optimization
We now describe how each agent𝑖 ∈𝐶 learns to identify the agent 𝑘∗∈𝐶\{𝑖}with the highest
observed reputation as defined in Eq. (??), and use the behavior of agent 𝑘∗to optimize its own
reputation. Designing an algorithm that is both scalable and efficient in helping agents to identify
an agent with the highest observed reputation is a key challenge of the overall learning algorithm,
as we explain below.
To identify an agent 𝑘∗ ∈𝐶\{𝑖}with the highest observed reputation, agents must (locally)
interact to exchange their estimates of the perceived benefits of others’ behavior. However, this
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 23

1079
1080
1081
1082
1083
1084
1085
1086
1087
1088
1089
1090
1091
1092
1093
1094
1095
1096
1097
1098
1099
1100
1101
1102
1103
1104
1105
1106
1107
1108
1109
1110
1111
1112
1113
1114
1115
1116
1117
1118
1119
1120
1121
1122
1123
1124
1125
1126
1127
Learning Common Norms in Multi-Agent Systems 23
local exchange of reputation estimates poses a scalability challenge. In large groups, agents may
need to communicate a substantial number of estimates, which can result in prohibitive overhead
in systems with limited communication capacity. To address this, we propose a novel algorithm in
which, at each time step, agents only exchange their estimate for the agent they currently believe
has the highest observed reputation. We formally show that this communication-efficient algorithm
still allows agents to correctly identify the highest-reputation agent with probability 1.
More precisely, we use an approach with the following two key components. First, each agent
keeps locally an estimate of the personal benefit they receive from the behavior of other agents.
Second, agents interact with each other and exchange estimates of their personal benefit in order
to obtain an estimate of the benefit of an agent’s behavior to the overall group. This estimate of the
the benefit of an agent’s behavior to the overall group is equivalent to the observed reputation of
an agent.
6.4.1 Local Information. To define the algorithm, we first define the information that each agent
∈𝐶 keeps locally.
Recall that the reputation of an agent 𝑘 ∈𝐶 is defined as:
𝑅𝑘(𝜋𝑘,𝜇𝑘)=
∑︁
𝑖∈𝐶\{𝑘}
𝜃 (𝜇𝑘)𝑈𝑖(𝜋𝑘),
where 𝜋𝑘 is the norm used by agent 𝑘, 𝑈𝑖(𝜋𝑘)is the utility agent 𝑖derives from it, and 𝜃 (𝜇𝑘)is the
probability that agent 𝑘 is active in group at time 𝑡.
Each agent 𝑖 ∈𝐶maintains the following time-dependent local estimates at each time step𝑡 ∈N:
(1) Personal Benefit Estimates 𝑣𝑖(𝑘,𝑡): an estimate of the benefit 𝜃 (𝜇𝑘(𝑡))𝑈𝑖(𝜋𝑘(𝑡))that agent
𝑘 ≠ 𝑖’s norm provides to agent 𝑖.
(2) Observed Reputation Estimates 𝑠𝑖(𝑘,𝑡): an estimate of the reputation 𝑅𝑘(𝜋𝑘(𝑡),𝜇𝑘(𝑡))of
agent 𝑘 ≠ 𝑖.
(3) Estimate of Agent with the Highest Observed Reputtion: Each agent 𝑖keeps a estimate 𝐿𝑖(𝑡)
of which agent they believe is the agent with the highest observed reputation.
We now define how each of these local estimates are initialized and updated over time.
6.4.2 Learning Personal Benefit. Each agent 𝑖initializes its estimate of the personal benefit of other
agents’ behavior by setting
𝑣𝑖(𝑘,0)= 0 𝑘 ∈𝐶\{𝑖}.
Each agent 𝑖updates its personal benefit estimates for all other agents that 𝑘 ∈A𝑎(𝑡)that are
active in the group at time 𝑡 by setting
𝑣𝑖(𝑘,𝑡 +1)= [𝑣𝑖(𝑘,𝑡)+𝜂𝑣(𝑡)[𝑢𝑖(𝑠(𝑡),𝑥𝑘(𝑡))− 𝑣𝑖(𝑘,𝑡)]], 𝑘 ∈𝐶\{𝑖},
where 𝑠(𝑡)is the state at time𝑡, 𝑥(𝑡)is the action that agent𝑘takes at time 𝑡, and 𝜂𝑣(𝑡)is a stepsize
parameter.
Furthermore, each agent 𝑖updates its estimate for all agents 𝑘 ∉ A(𝑡)that are not active at time
𝑡 by setting
𝑢𝑖(𝑠(𝑡),𝑥𝑘(𝑡))= 0
and
𝑣𝑖(𝑘,𝑡 +1)= [𝑣𝑖(𝑘,𝑡)+𝜂𝑣(𝑡)[−𝑣𝑖(𝑘,𝑡)]], 𝑘 ∈𝐶\{𝑖}.
Note that each agent 𝑖 ∈A updates its estimates in this way whether they are active or not in
the group at time 𝑡.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 24

1128
1129
1130
1131
1132
1133
1134
1135
1136
1137
1138
1139
1140
1141
1142
1143
1144
1145
1146
1147
1148
1149
1150
1151
1152
1153
1154
1155
1156
1157
1158
1159
1160
1161
1162
1163
1164
1165
1166
1167
1168
1169
1170
1171
1172
1173
1174
1175
1176
24 Vedic Sharma and Peter Marbach
6.4.3 Learning Observed Reputation. Each agent 𝑖initializes its estimate of the observed reputation
of other agents’ behavior by setting
𝑠𝑖(𝑘,0)= 0 for all 𝑘 ∈𝐶\{𝑖}.
All agents𝑖 ∈A𝑝(𝑡)that are active as participants at time𝑡, interact and exchange their estimates
of the observed reputation of other agents.
Recall that under the interaction rate 𝜇𝑝,𝑖(𝑡), agent 𝑖 is active as a participant in the group with
probability 𝜃  𝜇𝑝,𝑖
. In the following we define how the agents in A𝑝(𝑡)interact in order to update
their estimates of the agent with the highest observed reputation. To do that, we first define the
information that they exchange, and then define how the two agents use this information to update
their estimates of the observed reputation.
Estimate Update. Agents 𝑖 ∈A𝑝(𝑡)update their reputation estimates {𝑠𝑖(𝑘,𝑡)}𝑘∈B(𝑡)\𝑖 by taking
the average estimates over all agents 𝑗 ∈A(𝑡), and setting
𝑠𝑖(𝑘,𝑡 +1)=
Í
𝑗∈B(𝑡)\𝑘𝑠𝑗(𝑘,𝑡)
|A(𝑡)| +𝑣𝑖(𝑘,𝑡 +1)−𝑣𝑖(𝑘,𝑡), 𝑘 ∈B(𝑡)\𝑖. (9)
If agent 𝑖is not active at time 𝑡, and 𝑖 ∉ A(𝑡), then its estimates for all other agents 𝑘 ≠ 𝑖remain
unchanged, and we set
𝑠𝑖(𝑘,𝑡 +1)= 𝑠𝑖(𝑘,𝑡).
6.4.4 Identifying the Highest-Reputation Agent.All agents 𝑖 ∈A𝑝(𝑡)that are active as a participant
at time 𝑡 update their estimate of the agent with the highest observed reputation as follows.
Intuitively, agent’s 𝑖estimate of the agent with the highest observed observation at time 𝑡+1,
is the agent with the highest reputation estimate 𝑠𝑖(𝑘,𝑡 +1). However, in order to handle noise
in reputation estimates, agents choose the preferred agent employing a tie threshold Δ > 0: if
multiple agents’ reputations fall withinΔ of the highest estimate, the follower selects one uniformly
at random (avoiding overfitting to transient fluctuations). This promotes robustness in dynamic
environments. Formally, let Δ > 0 be a tie threshold and let
K𝑖(𝑡+1)=

𝑘 ∈𝐶\{𝑖}: 𝑠𝑖(𝑘,𝑡 +1)≥ max
𝑘′≠𝑖
𝑠𝑖(𝑘′,𝑡 +1)−Δ

.
Then the estimate of the agent with the highest observed reputation 𝐿𝑖(𝑡+1)is chosen uniformly
at random from the set K𝑖(𝑡+1).
If agent 𝑖is not active at time 𝑡, and 𝑖 ∉ A(𝑡), then its estimates for all other agents 𝑘 ≠ 𝑖remain
unchanged, and we set
𝐿𝑖(𝑡+1)= 𝐿𝑖(𝑡).
6.4.5 Optimizing Reputation. To optimize its reputation, agent 𝑖follows the agent 𝑘 identified by
𝐿𝑖(𝑡+1)and adopts 𝑘’s behavior directly. Formally, this means
𝜙(𝑤𝑟
𝑖 (𝑡+1);𝑠,𝑥)= 𝜙(𝑤𝑘(𝑡);𝑠,𝑥), 𝑠 ∈S, 𝑥 ∈X𝑠.
where 𝑤𝑘(𝑡)are the weights that agent 𝑘 uses at time 𝑡 to determine its behavior, i.e. which action
𝑥 to choose in state 𝑠(𝑡). Concretely, if agent 𝑘 optimizes its personal utility at time 𝑡 and 𝑘 ∈P(𝑡),
then we have that
𝑤𝑘(𝑡)= 𝑤𝑝𝑢
𝑘 (𝑡),
otherwise we have that
𝑤𝑘(𝑡)= 𝑤𝑠
𝑘(𝑡),
This emulation allows agent 𝑖 to acquire the reputation benefits associated with agent 𝑘’s norm.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 25

1177
1178
1179
1180
1181
1182
1183
1184
1185
1186
1187
1188
1189
1190
1191
1192
1193
1194
1195
1196
1197
1198
1199
1200
1201
1202
1203
1204
1205
1206
1207
1208
1209
1210
1211
1212
1213
1214
1215
1216
1217
1218
1219
1220
1221
1222
1223
1224
1225
Learning Common Norms in Multi-Agent Systems 25
6.5 Optimizing Status
Finally, we define how an agent 𝑖 ∈A𝑎(𝑡)that is active as an actor in the group at time 𝑡 updates
its weights 𝑤𝑠
𝑖 (𝑡)to maximize its status.
6.5.1 Objective Function. In this case, at time 𝑡, agent 𝑘 seeks to choose the parameter 𝑤𝑠
𝑘 that
maximizes its expected social support from followers F𝑘(𝑡)across states and actions:
𝐽𝑠(𝑤𝑠
𝑘,𝑡)=
∑︁
𝑠∈𝑆
𝑝(𝑠)
∑︁
𝑥∈X𝑠
𝜋(𝑤𝑠
𝑘;𝑠,𝑥)
∑︁
𝑖∈F𝑘 (𝑡)
𝜃 (𝜇𝑖(𝑡))𝑢𝑘(𝑠,𝑥). (10)
6.5.2 Instantaneous Social Support. In the following, agent’s use the social support obtained at
time 𝑡 to estimate the objective function 𝐽𝑠(𝑤𝑠
𝑘,𝑡). We refer to this support as the instantaneous
social support received by agent𝑖at time 𝑡. This instantaneous social support is a noisy observation
of the actual social support given by 𝐽(𝑤𝑠
𝑘,𝑡).
Formally, when 𝑘 chooses action 𝑥(𝑡)in state 𝑠(𝑡), the instantaneous social support that agent 𝑘
receives at time 𝑡 from a follower 𝑖 ∈F𝑘(𝑡)is given by
𝐷𝑖,𝑘(𝑠(𝑡),𝑥(𝑡),𝑡)= 𝐼𝑖(𝑡)𝑢𝑖(𝑠(𝑡),𝑥(𝑡)), (11)
where 𝐼𝑖(𝑡)is the indicator whether agent 𝑖is active in the group at time 𝑡 as an participant, i.e.,
whether 𝑖 ∈A𝑝(𝑡).
The total instantaneous social support received for the action action 𝑥(𝑡)in state 𝑠(𝑡)at time 𝑡 is
then given by ∑︁
𝑖∈F𝑘 (𝑡)
𝐷𝑖,𝑘(𝑠(𝑡),𝑥(𝑡),𝑡),
6.5.3 Gradient Estimation and Update. Using the total instantaneous support, agent 𝑘 then uses
again a standard stochastic gradient algorithm to maximize the objective function 𝐽(𝑤𝑘,𝑡). That is,
given the state 𝑠(𝑡)at time 𝑡, agent 𝑘first chooses an action 𝑥𝑘(𝑡)where the action 𝑥𝑘(𝑡)randomly
according to the random policy 𝜋(𝑤𝑠
𝑘(𝑡); 𝑠(𝑡),𝑥)for the current weight vector 𝑤𝑠
𝑘(𝑡). Next, agent 𝑖
updates its weight vector by setting
𝑤𝑠
𝑘(𝑡+1)= 𝑤𝑠
𝑘(𝑡)+𝛽(𝑡)·∇ 𝑤 log 𝜋(𝑤𝑠
𝑘(𝑡);𝑠(𝑡),𝑥𝑘(𝑡))·
∑︁
𝑖∈𝐹𝑘 (𝑡)
𝐷𝑖,𝑘(𝑠(𝑡),𝑥𝑘(𝑡),𝑡), (12)
where 𝛽(𝑡)is a stepsize parameter.
This learning rule increases the likelihood of actions that elicit stronger positive responses (i.e.,
higher utility) from followers, thereby encouraging norms that are broadly beneficial and socially
rewarded.
6.6 Updating Reward Estimates
Let ˆ𝐽𝑝𝑢
𝑖 (𝑡), ˆ𝐽𝑟
𝑖 (𝑡), and ˆ𝐽𝑠
𝑖 (𝑡), be the estimates of the rewards obtained in the different roles. Agent
𝑖 initializes these estimates at time 𝑡 = 0 by setting them equal to 0, and then uses the learning
algorithm provided in this section to update these estimates at time 𝑡 ≥1.
Moreover, agent 𝑖will use these estimates both to decide which role to take on, as well as decide
on its interaction rates 𝜇𝑎,𝑖(𝑡)and 𝜆𝑎,𝑖(𝑡).
Updating Estimate of Personal Utility. All agents such that 𝑖 ∈P(𝑡)and 𝑖 ∈A𝑎(𝑡)that are active
as an actor at time 𝑡 and optimize their personal utility at time 𝑡 use the observed personal benefit
𝑢𝑖(𝑠(𝑡),𝑥𝑖(𝑡))
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 26

1226
1227
1228
1229
1230
1231
1232
1233
1234
1235
1236
1237
1238
1239
1240
1241
1242
1243
1244
1245
1246
1247
1248
1249
1250
1251
1252
1253
1254
1255
1256
1257
1258
1259
1260
1261
1262
1263
1264
1265
1266
1267
1268
1269
1270
1271
1272
1273
1274
26 Vedic Sharma and Peter Marbach
to update their estimate of the reward for personal utility by setting
ˆ𝐽𝑝𝑢
𝑖 (𝑡)= ˆ𝐽𝑝𝑢
𝑖 (𝑡−1)+𝜂𝐽(𝑡)

𝑢𝑖(𝑠(𝑡),𝑥𝑖(𝑡))− ˆ𝐽𝑝𝑢
𝑖 (𝑡−1)

where 𝜂𝐽(𝑡)is a stepsize parameter.
Updating Estimate of Reward from Reputation. All agents 𝑖 ∈R(𝑡)and 𝑖 ∈A𝑎(𝑡)that are active
as an actor at time 𝑡 and that optimize their reputation at time 𝑡 update their estimate as
ˆ𝐽𝑟
𝑖 (𝑡)= 𝑠𝑖(𝑘,𝑡),
where 𝑘 is the agent that 𝑖follows in order to optimize its reputation.
Updating Estimate of Reward from Status. All agents 𝑘 ∈A𝑎(𝑡)that are active as an actor at time
𝑡 and have followers at time 𝑡 (F𝑘(𝑡)≠ ∅) use the instantaneous social support
∑︁
𝑗∈F𝑘 (𝑡)
𝐷𝑗,𝑘(𝑠(𝑡),𝑥(𝑡),𝑡)
to update their estimate of the reward for status by setting
ˆ𝐽𝑠
𝑘(𝑡)= ˆ𝐽𝑠
𝑘(𝑡−1)+𝜂𝐽(𝑡)

∑︁
𝑗∈F𝑘 (𝑡)
𝐷𝑗,𝑘(𝑠(𝑡),𝑥(𝑡),𝑡)− ˆ𝐽𝑠
𝑘(𝑡−1)

,
where 𝜂𝐽(𝑡)is a stepsize parameter.
Unchanged Estimates. All other estimates that are not updated according to the above rules
remain unchanged; that is, their value at time 𝑡 equals their value at time 𝑡−1.
Discussion. Note that while most agents update only a single reward estimate at each time 𝑡,
there is one important exception: agents that optimize their personal utility and have followers
(P(𝑡)∋ 𝑖and F𝑖(𝑡)≠ ∅) update both their estimate for personal utility and their estimate for status.
This reflects the fact that such agents not only observe their own direct utility but also receive
social feedback from followers, allowing them to refine both estimates simultaneously.
6.7 Optimizing Interaction Rates as an Actor
Let
ˆ𝐻𝑖(𝑡)= max{ˆ𝐽𝑝𝑢
𝑖 (𝑡),𝛾 ˆ𝐽𝑟
𝑖 (𝑡),𝜅 ˆ𝐽𝑠
𝑖 (𝑡)}.
We set
𝜇𝑎,𝑖(𝑡)=

𝜇𝑎,𝑖(𝑡−1)+𝛼(𝑡)

−𝜃′ 𝑀−𝜇𝑎,𝑖(𝑡−1)𝑢0 +𝜃′ 𝜇𝑎,𝑖(𝑡−1) ˆ𝐻𝑖(𝑡)
 𝑀
0 , (13)
where 𝑢0 > 0 is a given constant that reflects the utitility that an agent obtains from participating
as an actor outside the group, and
[𝑥]𝑀
0 = min{max(𝑥,0),𝑀}, 𝑥 ∈R.
6.8 Discussion
The algorithm we propose presents several challenges. First, it introduces a novel reputation
learning process that runs in parallel with the agents’ reward optimization algorithms. Second,
agents can switch roles, and each role corresponds to a different reward component. As a result,
the algorithm does not optimize a continuous objective function: each role switch introduces a
“jump” or discontinuity in the objective.
To address these challenges and keep the analysis and notation tractable - which are already
non-trivial even with simplifying assumptions - we adopt a few simplifications. Importantly, these
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 27

1275
1276
1277
1278
1279
1280
1281
1282
1283
1284
1285
1286
1287
1288
1289
1290
1291
1292
1293
1294
1295
1296
1297
1298
1299
1300
1301
1302
1303
1304
1305
1306
1307
1308
1309
1310
1311
1312
1313
1314
1315
1316
1317
1318
1319
1320
1321
1322
1323
Learning Common Norms in Multi-Agent Systems 27
assumptions can be relaxed without affecting our main results. In the following, we describe these
assumptions and discuss how they could be lifted.
6.8.1 Reputation Learning. To simplify both the notation and the analysis, we assume in this paper
that all agents 𝑖 ∈A(𝑡)that are active at time 𝑡 participate in the reputation learning algorithm
defined in Subsection 6.3.2. However, this assumption can be relaxed in two important ways:
(1) Random subset interactions: Instead of assuming that all active agents interact at each
time step, we can consider the case where only a random subset of agents𝑖 ∈A(𝑡)engage in
interactions to exchange their estimates of which agent has the highest observed reputation.
(2) Connectivity graph: We can assume that agents interact according to an underlying
connectivity graph G(V,E), where the nodes Vcorrespond to the set of agents A, and E
defines the edges between agents. At time 𝑡, two agents 𝑖 and 𝑗 exchange their estimates
only if both are active (𝑖,𝑗 ∈A(𝑡)) and there exists an edge (𝑖,𝑗 )∈E in the graph G.
Our main results continue to hold under either of these relaxed assumptions, provided that:
•In the first case, each active agent 𝑖 ∈A(𝑡)has a probability of at least 𝑝0 > 0 of being
selected for interaction at each time step.
•In the second case, the connectivity graph Gis connected (i.e., there exists a path between
any pair of agents).
Both of these conditions are mild and align with standard assumptions in the distributed learning
literature. They ensure that information about the highest reputation agent eventually propagates
throughout the population. Importantly, relaxing the assumption that all active agents participate
at each step does not affect the convergence guarantees of the reputation learning algorithm or the
main results of our analysis.
6.8.2 Reputation Optimization. The approach to reputation optimization presented in this work
intentionally adopts the simplest possible mechanism: once an agent 𝑖identifies another agent 𝑘 as
having the highest reputation, it directly copies 𝑘’s policy parameters. This hard imitation ensures
that agent 𝑖inherits the same norm as 𝑘, and thereby gains similar reputational benefits.
However, more sophisticated reputation learning schemes could be employed. For example,
instead of directly copying the policy parameters of agent𝑘, agent 𝑖could observe the actions taken
by 𝑘 across a sequence of states and train its own policy representation to mimic this behavior. In
this approach, agent𝑖would maintain a dataset of state-action pairs(𝑠,𝑥𝑘)based on observations of
agent 𝑘, and update its own parameters via supervised learning to minimize a divergence between
its policy and the observed behavior:
min
𝑤𝑖
∑︁
(𝑠,𝑥𝑘 )
−log 𝜋(𝑤𝑖;𝑠,𝑥𝑘).
This imitation learning framework, often referred to as behavioral cloning, enables smoother
adaptation and allows for partial or noisy observations. While we do not implement this in our
current model, such techniques could enhance robustness and learning efficiency in more complex
or partially observable environments.
Importantly, such a learning-based imitation scheme can be easily incorporated into our algorithm
and analysis using standard techniques from online learning and policy optimization. However, to
keep the exposition and notation streamlined-and to avoid additional overhead in the theoretical
analysis-we adopt the simpler direct-copying rule in this work. This choice allows us to highlight
the essential structure and convergence properties of the overall system without distraction from
secondary implementation details.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 28

1324
1325
1326
1327
1328
1329
1330
1331
1332
1333
1334
1335
1336
1337
1338
1339
1340
1341
1342
1343
1344
1345
1346
1347
1348
1349
1350
1351
1352
1353
1354
1355
1356
1357
1358
1359
1360
1361
1362
1363
1364
1365
1366
1367
1368
1369
1370
1371
1372
28 Vedic Sharma and Peter Marbach
7 UPDATING ROLES
In this section we define how agents decide which role to take on. This process mirrors the role
decision rules given in Section ??, but here agents use estimates ˆ𝐽𝑝𝑢
𝑖 (𝑡), ˆ𝐽𝑟
𝑖 (𝑡), and ˆ𝐽𝑠
𝑖 (𝑡)of the
payoffs under different reward components, rather than the actual payoffs.
7.1 Definitions and Notation
In this subsection we introduce additional definitions and notation that we use in this section.
7.1.1 Roles and Followers. Recall that P(𝑡), R(𝑡), and S(𝑡), indicate which agents optimize their
personal utility, reputation, and status, respectively, at time 𝑡. At each time step 𝑡, each agent
decides on a single reward component to optimize, and we have that these sets are disjoint with
P(𝑡)∪R(𝑡)∪S(𝑡)= A, 𝑡 ≥0.
All agents start out by optimizing their personal utility, and we have that
P(0)= A,
and R(0)= S(0)= ∅.
Also recall that F𝑘(𝑡)denotes the set of agents 𝑖 ∈R(𝑡)that follow agent 𝑘 at time 𝑡 to optimize
their reputation. Initially we have that
F𝑘(𝑡)= ∅, 𝑘 ∈A.
In the following, we define how the sets P(𝑡), R(𝑡), and S(𝑡), and the sets F𝑘(𝑡), 𝑘 ∈A, evolve
over time as agents decide on which role to take on.
7.1.2 Threshold on Number of Followers.An agent considers optimizing its status only if it has
sufficiently many followers:
|F𝑖(𝑡)|≥ 𝑐𝑁,
where 𝑐 ∈[0,1]is a given constant.
This ensures that status decisions are based on reliable social feedback. With too few followers,
feedback may misrepresent community preferences, leading to actions that reduce status. Requiring
a larger base of followers increases the chance that behavior changes genuinely enhance both social
welfare and the agent’s status.
7.1.3 Threshold on Reputation. We introduce two reputation thresholds, 𝐵𝑅 and 𝐵𝐹, satisfying
0 < 𝐵𝐹 < 𝐵𝑅,
that agents use when deciding whether to follow another agent. Specifically:
a) An agent will decide to follow another agent only if that agent’s reputation is at least 𝐵𝑅.
b) Once an agent is a follower, it will continue following as long as the reputation of the agent
with the highest reputation remains above 𝐵𝐹.
This assumption creates a hysteresis: it takes a sufficiently high reputation to start following,
but a larger drop before an agent will stop following. This prevents agents from switching back
and forth when the highest reputation oscillates around 𝐵𝑅.
7.1.4 Update Epochs. Agents periodically re-evaluate which role to adopt. Let 𝑇𝑛 be a sequence of
positive integers indicating the interval between updates. For simplicity, we assume all agents use
the same sequence {𝑇𝑛}𝑛≥1, though this can be relaxed (see Subsection ??).
Setting 𝑠0 = 0, the 𝑛th update epoch is
𝑠𝑛 = 𝑠𝑛−1 +𝑇𝑛, 𝑛 ≥1.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 29

1373
1374
1375
1376
1377
1378
1379
1380
1381
1382
1383
1384
1385
1386
1387
1388
1389
1390
1391
1392
1393
1394
1395
1396
1397
1398
1399
1400
1401
1402
1403
1404
1405
1406
1407
1408
1409
1410
1411
1412
1413
1414
1415
1416
1417
1418
1419
1420
1421
Learning Common Norms in Multi-Agent Systems 29
At epoch 𝑠𝑛, each agent 𝑖 decides which reward component to optimize during the interval
𝑡 = 𝑠𝑛 +1,...,𝑠 𝑛+1 based on its payoff estimates.
7.2 Sequential Role Update Procedure
At each epoch 𝑠𝑛, all agents simultaneously decide which role to adopt. Since these decisions are
interdependent-especially for reputation optimization-the updates must be applied in a carefully
chosen order to avoid inconsistencies.
For example, if agent 𝑖 considers following agent 𝑗, but 𝑗 is already following agent 𝑘, then 𝑖
should ultimately follow 𝑘 rather than 𝑗. Proper ordering ensures that such indirect relationships
are respected: each agent observes the most up-to-date follower sets when making its decision.
The procedure updates roles sequentially, with each step using the role sets P, R, S, and follower
sets F𝑘 produced by all previous steps. Formally:
(1) Step 0 - Initialization: Copy the current role and follower sets:
P(𝑠𝑛)= P(𝑠𝑛−1), R(𝑠𝑛)= R(𝑠𝑛−1), S(𝑠𝑛)= S(𝑠𝑛−1), F𝑘(𝑠𝑛)= F𝑘(𝑠𝑛−1), ∀𝑘 ∈A.
(2) Step 1 - Reputation: Agents without followers decide whether to optimize reputation starting
at 𝑠𝑛 +1. This updates P(𝑠𝑛), R(𝑠𝑛), and F𝑘(𝑠𝑛).
(3) Step 2 - Status: Agents with sufficiently many followers and favorable status estimates
update their roles, modifying S(𝑠𝑛)and P(𝑠𝑛).
(4) Step 3 - Personal Utility: All remaining agents default to personal-utility optimization. This
finalizes P(𝑠𝑛), R(𝑠𝑛), S(𝑠𝑛), and F𝑘(𝑠𝑛).
This sequential, in-place procedure ensures consistency and eliminates ambiguities from indirect
follower relationships. We now define each of the three steps in detail.
7.3 Step 1 - Optimizing Reputation
At time 𝑡 = 𝑠𝑛, agents without followers evaluate whether their estimated reward from optimizing
reputation, 𝛾ˆ𝐽𝑟
𝑖 (𝑠𝑛), exceeds the reward from personal utility, ˆ𝐽𝑝𝑢
𝑖 (𝑠𝑛). If so, the agent follows the
one it believes currently has the highest reputation. To maintain consistent follower relationships,
agents never follow someone who is already a follower. The decision also uses one of two reputation
thresholds: 𝐵𝑅 for agents not yet following anyone, and 𝐵𝐹 for agents who are already followers,
creating a hysteresis effect that stabilizes following behavior as discussed above. The precise update
procedure is as follows.
Let C(𝑠𝑛)be the set of agents that do not have a follower at time 𝑡 = 𝑠𝑛. We further partition it
into C𝑟(𝑠𝑛), the agents already following another agent, and C𝑝𝑢(𝑠𝑛), the agents not yet following
anyone.
For each agent 𝑖 ∈C(𝑠𝑛), define the effective threshold
𝐵𝑖 =
(
𝐵𝐹, 𝑖 ∈C𝑟(𝑠𝑛)
𝐵𝑅, 𝑖 ∈C𝑝𝑢(𝑠𝑛)
to decide whether to follow or remain following.
Let 𝑁𝑐 = |C(𝑠𝑛)|, and let 𝜎 = (𝜎(1),...,𝜎 (𝑁𝑐))be a permutation representing the update order.
For 𝑞= 1,...,𝑁 𝑐, let 𝑖 = 𝜎(𝑞)and perform the following updates:
(1) If
𝛾ˆ𝐽𝑟
𝑖 (𝑠𝑛)> max{𝐵𝑖, ˆ𝐽𝑝𝑢
𝑖 (𝑠𝑛)},
then agent 𝑖 adopts reputation optimization for 𝑡 = 𝑠𝑛 +1,...,𝑠 𝑛+1 by following
ˆ𝑘 = 𝐿𝑖(𝑠𝑛),
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 30

1422
1423
1424
1425
1426
1427
1428
1429
1430
1431
1432
1433
1434
1435
1436
1437
1438
1439
1440
1441
1442
1443
1444
1445
1446
1447
1448
1449
1450
1451
1452
1453
1454
1455
1456
1457
1458
1459
1460
1461
1462
1463
1464
1465
1466
1467
1468
1469
1470
30 Vedic Sharma and Peter Marbach
the agent that 𝑖believes has the highest observed reputation.
(2) If ˆ𝑘 ∈F𝑘′(𝑠𝑛)for some 𝑘′, then agent 𝑖 follows 𝑘′and we set 𝑘 = 𝑘′. Otherwise, 𝑘 = ˆ𝑘.
(3) Update the role sets:
R(𝑠𝑛)= R(𝑠𝑛)∪{𝑖}, P(𝑠𝑛)= P(𝑠𝑛)\{𝑖}.
(4) Update the follower sets:
F𝑘(𝑠𝑛)= F𝑘(𝑠𝑛)∪{𝑖}, F𝑗(𝑠𝑛)= F𝑗(𝑠𝑛)\{𝑖}, ∀𝑗 ≠ 𝑘.
This procedure ensures that all agents without followers are updated sequentially and never
follow an agent who is already a follower, preserving the consistency of indirect follower relation-
ships.
7.4 Step 2 - Optimizing Status
At time 𝑡 = 𝑠𝑛, if agent 𝑖 has a sufficiently large number of followers,
|F𝑖(𝑠𝑛)|≥ 𝑐𝑁,
and its estimated reward from optimizing status exceeds that from personal utility, i.e.,
𝜅ˆ𝐽𝑠
𝑖 (𝑠𝑛)> ˆ𝐽𝑝𝑢
𝑖 (𝑠𝑛),
then agent 𝑖will optimize its status during 𝑡 = 𝑠𝑛 +1,...,𝑠 𝑛+1. The role sets are updated as:
S(𝑠𝑛)= S(𝑠𝑛)∪{𝑖}, P(𝑠𝑛)= P(𝑠𝑛)\{𝑖}.
7.5 Step 3 - Optimizing Personal Utility
If agent 𝑖does not meet the conditions to optimize reputation or status, it will optimize its personal
utility during 𝑡 = 𝑠𝑛 +1,...,𝑠 𝑛+1. We update:
P(𝑠𝑛)= P(𝑠𝑛)∪{𝑖},
R(𝑠𝑛)= R(𝑠𝑛)\{𝑖}, S(𝑠𝑛)= S(𝑠𝑛)\{𝑖}.
For all agents 𝑗 ≠ 𝑖we set
F𝑗(𝑠𝑛)= F𝑗(𝑠𝑛)\{𝑖}.
7.6 Discussion
The threshold parameter 𝑐, which governs the decision to switch from optimizing personal utility
to optimizing status, plays a crucial role. If 𝑐 is set too low, agents may switch prematurely to
optimizing status, leading to reduced reputation and a potential loss of followers. If 𝑐 is too high,
agents may delay switching even when status optimization would be beneficial.
The appropriate value of 𝑐 depends on factors such as group size 𝑁 and the personal utility
functions 𝑢𝑖(𝑠,𝑥)that agents obtain from different states and actions. As such, the optimal value of
𝑐 cannot be known in advance and must be learned from social feedback. Specifically, if an agent
switches to optimizing status and subsequently experiences a decline in its number of followers,
this indicates that the agent’s contribution may no longer align with the social reward structure.
In such cases, the agent should revert to optimizing personal utility and increase the threshold 𝑐,
making it less likely to switch to status optimization prematurely in the future. Conversely, if the
agent’s follower count remains stable or increases, the current value of𝑐can be maintained or even
slightly decreased. This adaptive process allows agents to calibrate role transitions dynamically
in response to their social environment, fostering a balance between self-interest and socially
beneficial behavior.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 31

1471
1472
1473
1474
1475
1476
1477
1478
1479
1480
1481
1482
1483
1484
1485
1486
1487
1488
1489
1490
1491
1492
1493
1494
1495
1496
1497
1498
1499
1500
1501
1502
1503
1504
1505
1506
1507
1508
1509
1510
1511
1512
1513
1514
1515
1516
1517
1518
1519
Learning Common Norms in Multi-Agent Systems 31
Although incorporating learning dynamics for the threshold parameter𝑐is possible, we omit this
analysis to avoid overloading the algorithm definition and theoretical analysis, which are already
complex. We leave this extension for future work.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 32

1520
1521
1522
1523
1524
1525
1526
1527
1528
1529
1530
1531
1532
1533
1534
1535
1536
1537
1538
1539
1540
1541
1542
1543
1544
1545
1546
1547
1548
1549
1550
1551
1552
1553
1554
1555
1556
1557
1558
1559
1560
1561
1562
1563
1564
1565
1566
1567
1568
32 Vedic Sharma and Peter Marbach
8 STEPSIZE PARAMETERS, UPDATE INTERVALS, AND TIME-SCALES
The convergence and stability of the learning dynamics in our model critically depend on the
appropriate choice of stepsize parameters and the careful design of a time-scale hierarchy across the
different update processes. Because agents simultaneously optimize multiple objectives-including
personal utility, reputation, and status-while also adapting their interaction rates and roles, en-
suring that these learning processes evolve on appropriately separated time-scales is essential for
preventing destabilizing feedback loops and guaranteeing convergence to a stable equilibrium.
In this section, we formally introduce the stepsize parameters associated with each learning
process and specify the assumptions required for their convergence. We begin by defining the time-
scale hierarchy, outlining the ordering of the different processes based on their update frequencies.
We then state the assumptions on the stepsize sequences and update intervals that enforce these
separations in time-scales. Together, these conditions ensure that faster processes, such as reputation
learning, stabilize before slower processes, such as behavior optimization and role adaptation, make
significant updates-thereby enabling convergence of the overall coupled learning system.
8.1 Two Time-Scale Approach
The learning dynamics in our model unfold across two main categories of processes given as
follows:
(1) Continuous Processes: These are processes in which agents make updates whenever they are
active in the group. They include the gossip proces, the process for updating the agents’
behavior, and the process for updating theinteraction rates.
(2) Discrete Processes: These are processes where updates occur only periodically, rather than
every time an agent is active. This include the processes for updating the agents’ roles.
To ensure a convergence to a learning outcome, these processes need to evolve on different time-
scaales: some processes need to evolve on a fast time-scales, allowing agents to quickly adapt to
new feedback, while other processes need to evolve on a slow time-scale, enabling more stable
structural adaptation and role differentiation. The positions of the different processes with respect
to these two time-scales are given as follows:
(1) Reputation Learning and Reward Estimation (Continuous Process on Fast Time-Scale): Reputa-
tion learning as defined in Section ?? and reward estimation as defined in Section ??, occur
on the fast time-scale. This includes estimating the personal benefit that agent 𝑖 receives
from another agent 𝑘, and identifying the agent with the highest observed reputation, and
updating the reward estimates for the different roles. The assumption that these processes
evolves on the fast time-scale is formalized in Assumption ?? and Assumption ??.
(2) Behavior Optimization (Continuous Process on Slow Time-Scale): Behavior optimization,
described in Sections ?? and ??, occurs on a slow time-scale. This ensures that updates of
the weights using gradient estimates tracks an ODE. The separation of this time-scale is
ensured by Assumption ??.
(3) Interaction Rate Optimization (Continuous Process on Slow Time-Scale): Interaction rate
optimization as defined in Section ??, occur on a slow time-scale. This ensure that the rates
do not change drastically between role updates. The separation of this time-scale is ensured
by Assumption ??.
(4) Role Optimization (Discrete Process on Slow Time-Scale): Finally, role optimization as defined
in Section ?? occurs on a slow time-scale. This process involves determining whether to
optimize personal utility, reputation, or status, and they only change after an accurate
estimate of the rewards that are obtained for each role is accumulated. The sufficiently slow
evolution of these processes is enforced by Assumptions ?? and ??.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 33

1569
1570
1571
1572
1573
1574
1575
1576
1577
1578
1579
1580
1581
1582
1583
1584
1585
1586
1587
1588
1589
1590
1591
1592
1593
1594
1595
1596
1597
1598
1599
1600
1601
1602
1603
1604
1605
1606
1607
1608
1609
1610
1611
1612
1613
1614
1615
1616
1617
Learning Common Norms in Multi-Agent Systems 33
This separation of time-scales ensures that lower-level dynamics stabilize before higher-level
decisions are made, thereby enabling convergence in the presence of coupled learning processes.
8.2 Assumptions on Stepsize Parameters
Recall that 𝛼(𝑡)is the setpsize parameter used to update the weights𝑤𝑝𝑢
𝑖 (𝑡), 𝑖 ∈A, for the behavior
used to optimize the personal utility, 𝛽(𝑡)is the setpsize parameter used to update the weights
𝑤𝑠
𝑖 (𝑡), 𝑖 ∈A, for the behavior used to optimize the status, 𝜂𝑣(𝑡)is the setpsize parameter used to
update the estimates 𝑣𝑖(𝑘,𝑡)of the benefit that the behavior of agent 𝑘 provides agent 𝑖, and 𝜂𝐽(𝑡)
is the setpsize parameter that agents use to update their estimate of the personal utility ˆ𝐽𝑝𝑢
𝑖 (𝑡)and
the rewards from status ˆ𝐽𝑝𝑢
𝑖 (𝑡).
We then make the following standard assumption for these stepsize parameters.
Assumption 5. The following properties hold for the stepsize parameters.
(1) The stepsizes 𝛼(𝑡), 𝛼(𝑡)> 0, satisfy the conditions
∞∑︁
𝑡=1
𝛼(𝑡)= ∞, and
∞∑︁
𝑡=1
𝛼(𝑡)2 < ∞.
(2) The stepsizes 𝛽(𝑡), 𝛽(𝑡)> 0, satisfy the conditions
∞∑︁
𝑡=1
𝛽(𝑡)= ∞, and
∞∑︁
𝑡=1
𝛽(𝑡)2 < ∞.
(3) The stepsizes 𝜂𝑣(𝑡), 𝜂𝑣(𝑡)> 0, satisfy the conditions
∞∑︁
𝑡=1
𝜂𝑣(𝑡)= ∞, and
∞∑︁
𝑡=1
𝜂𝑣(𝑡)2 < ∞.
(4) The stepsizes 𝜂𝐽(𝑡), 𝜂𝐽(𝑡)> 0, satisfy the conditions
∞∑︁
𝑡=1
𝜂𝐽(𝑡)= ∞, and
∞∑︁
𝑡=1
𝜂𝐽(𝑡)2 < ∞.
8.3 Assumptions on Update Intervals
We make the following assumption for the update intervals𝑇𝑛 and 𝑇rate
𝑛 .
Assumption 6. For the update intervals {𝑇𝑛}𝑛≥1 we have that
lim
𝑛→∞
𝑇𝑛 = ∞,.
8.4 Assumptions on Time-Scales
The next assumption ensures that the behavior learning using the stepsiz parameters𝛼(𝑡)and 𝛼(𝑡)
evolves on a slower time-scale than estimating the personal benefit and the obtained rewards under
the different roles that uses the stepsize parameters 𝜂𝑣(𝑡)and 𝜂𝐽(𝑡).
Assumption 7. For the stepsizes 𝛼(𝑡), 𝛽(𝑡), 𝜂𝑣(𝑡), and 𝜂𝑣(𝑡), we have that
(1)
lim
𝑡→∞
𝛼(𝑡)/𝜂𝑣(𝑡)= 0
(2)
lim
𝑡→∞
𝛼(𝑡)/𝜂𝐽(𝑡)= 0
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 34

1618
1619
1620
1621
1622
1623
1624
1625
1626
1627
1628
1629
1630
1631
1632
1633
1634
1635
1636
1637
1638
1639
1640
1641
1642
1643
1644
1645
1646
1647
1648
1649
1650
1651
1652
1653
1654
1655
1656
1657
1658
1659
1660
1661
1662
1663
1664
1665
1666
34 Vedic Sharma and Peter Marbach
(3)
lim
𝑡→∞
𝛽(𝑡)/𝜂𝑣(𝑡)= 0
(4)
lim
𝑡→∞
𝛽(𝑡)/𝜂𝐽(𝑡)= 0
Our final assumption characterizes how the time-scale of update intervals 𝑇rate
𝑛 compare with
the time-scale of the stepsizes 𝛼(𝑡), 𝛽(𝑡), 𝜂𝑣(𝑡), and 𝜂𝑣(𝑡),
Assumption 8. For the stepsizes 𝛼(𝑡), 𝛽(𝑡), 𝜂𝑣(𝑡), and 𝜂𝑣(𝑡), we have for all agents 𝑖 ∈𝐶 that
(1)
lim
𝑛→∞
𝑠(𝑛+1)∑︁
𝑡=𝑠(𝑛)+1
𝜂𝑣(𝑡)= ∞.
(2)
lim
𝑛→∞
𝑠(𝑛+1)∑︁
𝑡=𝑠(𝑛)+1
𝜂𝐽(𝑡)= ∞.
(3)
lim
𝑛→∞
𝑠(𝑛+1)∑︁
𝑡=𝑠(𝑛)+1
𝛼(𝑡)= 0.
(4)
lim
𝑛→∞
𝑠(𝑛+1)∑︁
𝑡=𝑠(𝑛)+1
𝛽(𝑡)= 0.
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 35

1667
1668
1669
1670
1671
1672
1673
1674
1675
1676
1677
1678
1679
1680
1681
1682
1683
1684
1685
1686
1687
1688
1689
1690
1691
1692
1693
1694
1695
1696
1697
1698
1699
1700
1701
1702
1703
1704
1705
1706
1707
1708
1709
1710
1711
1712
1713
1714
1715
Learning Common Norms in Multi-Agent Systems 35
REFERENCES
[1] Agarwal, A., Kumar, S., and Sycara, K. Learning transferable cooperative behavior in multi-agent teams. arXiv
preprint arXiv:1906.01202 (2019).
[2] Amato, C. A first introduction to cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2405.06161
(2024).
[3] Amato, C.An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement
learning. arXiv preprint arXiv:2409.03052 (2024).
[4] Brekke, K. A., Øksendal, B., and Stenseth, N. C.The effect of climate variations on the dynamics of pasture-livestock
interactions under cooperative and noncooperative management. Proceedings of the National Academy of Sciences 104 ,
37 (2007), 14730-14734.
[5] Chen, W., Li, W., Liu, X., Yang, S., and Gao, Y. Learning explicit credit assignment for cooperative multi-agent
reinforcement learning via polarization policy gradient. In Proceedings of the AAAI Conference on Artificial Intelligence
(2023), vol. 37, pp. 11542-11550.
[6] Chu, T., Chinchali, S., and Katti, S. Multi-agent reinforcement learning for networked system control. arXiv
preprint arXiv:2004.01339 (2020).
[7] Claus, C., and Boutilier, C. The dynamics of reinforcement learning in cooperative multiagent systems. AAAI/IAAI
1998, 746-752 (1998), 2.
[8] Eccles, T., Bachrach, Y., Lever, G., Lazaridou, A., and Graepel, T. Biases for emergent communication in
multi-agent reinforcement learning. Advances in neural information processing systems 32 (2019).
[9] Foerster, J., Assael, I. A., De Freitas, N., and Whiteson, S. Learning to communicate with deep multi-agent
reinforcement learning. Advances in neural information processing systems 29 (2016).
[10] Goeckner, A., Sui, Y., Martinet, N., Li, X., and Zhu, Q. Graph neural network-based multi-agent reinforcement
learning for resilient distributed coordination of multi-robot systems. In 2024 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS) (2024), IEEE, pp. 5732-5739.
[11] Hardin, G. The tragedy of the commons: the population problem has no technical solution; it requires a fundamental
extension in morality. science 162 , 3859 (1968), 1243-1248.
[12] Hu, G., Zhu, Y., Zhao, D., Zhao, M., and Hao, J. Event-triggered multi-agent reinforcement learning with communi-
cation under limited-bandwidth constraint. arXiv preprint arXiv:2010.04978 (2020).
[13] Jaqes, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P., Strouse, D., Leibo, J. Z., and De Freitas, N.Social
influence as intrinsic motivation for multi-agent deep reinforcement learning. In International conference on machine
learning (2019), PMLR, pp. 3040-3049.
[14] Jiang, J., and Lu, Z. Learning attentional communication for multi-agent cooperation. Advances in neural information
processing systems 31 (2018).
[15] Kilinc, O., and Montana, G. Multi-agent deep reinforcement learning with extremely noisy observations. arXiv
preprint arXiv:1812.00922 (2018).
[16] Kim, W., Cho, M., and Sung, Y. Message-dropout: An efficient training method for multi-agent deep reinforcement
learning. In Proceedings of the AAAI conference on artificial intelligence (2019), vol. 33, pp. 6079-6086.
[17] Kumar, D., Baranwal, G., Raza, Z., and Vidyarthi, D. P.A systematic study of double auction mechanisms in cloud
computing. Journal of Systems and Software 125 (2017), 234-255.
[18] Lin, T., Huh, J., Stauffer, C., Lim, S. N., and Isola, P. Learning to ground multi-agent communication with
autoencoders. Advances in Neural Information Processing Systems 34 (2021), 15230-15242.
[19] Lowe, R., Wu, Y. I., Tamar, A., Harb, J., Pieter Abbeel, O., and Mordatch, I. Multi-agent actor-critic for mixed
cooperative-competitive environments. Advances in neural information processing systems 30 (2017).
[20] Lyu, X., Baisero, A., Xiao, Y., Daley, B., and Amato, C. On centralized critics in multi-agent reinforcement learning.
Journal of Artificial Intelligence Research 77 (2023), 295-354.
[21] Lyu, X., Xiao, Y., Daley, B., and Amato, C. Contrasting centralized and decentralized critics in multi-agent reinforce-
ment learning. arXiv preprint arXiv:2102.04402 (2021).
[22] Malysheva, A., Sung, T. T., Sohn, C.-B., Kudenko, D., and Shpilman, A.Deep multi-agent reinforcement learning
with relevance graphs. arXiv preprint arXiv:1811.12557 (2018).
[23] Nagpal, K., Dong, D., Bouvier, J.-B., and Mehr, N. Leveraging large language models for effective and explainable
multi-agent credit assignment. arXiv preprint arXiv:2502.16863 (2025).
[24] Ostrom, E. Governing the commons: The evolution of institutions for collective action . Cambridge university press, 1990.
[25] Prasad, A. S., and Rao, S. A mechanism design approach to resource procurement in cloud computing. IEEE
Transactions on Computers 63 , 1 (2013), 17-30.
[26] Qu, C., Li, H., Liu, C., Xiong, J., Chu, W., Wang, W., Qi, Y., Song, L., et al.Intention propagation for multi-agent
reinforcement learning.
[27] Singh, A., Jain, T., and Sukhbaatar, S. Learning when to communicate at scale in multiagent cooperative and
, Vol. 1, No. 1, Article . Publication date: February 2026.

---

## Page 36

1716
1717
1718
1719
1720
1721
1722
1723
1724
1725
1726
1727
1728
1729
1730
1731
1732
1733
1734
1735
1736
1737
1738
1739
1740
1741
1742
1743
1744
1745
1746
1747
1748
1749
1750
1751
1752
1753
1754
1755
1756
1757
1758
1759
1760
1761
1762
1763
1764
36 Vedic Sharma and Peter Marbach
competitive tasks. arXiv preprint arXiv:1812.09755 (2018).
[28] Sukhbaatar, S., Fergus, R., et al. Learning multiagent communication with backpropagation. Advances in neural
information processing systems 29 (2016).
[29] Vickrey, W. Counterspeculation, auctions, and competitive sealed tenders. The Journal of finance 16 , 1 (1961), 8-37.
[30] W ang, C., and Durugkar, I.Dm2: Decentralized multi-agent reinforcement learning via distribution. In Proceedings
of the 37th Conference on Artificial Intelligence (2023).
[31] Wen, G., Fu, J., Dai, P., and Zhou, J. Dtde: A new cooperative multi-agent reinforcement learning framework. The
Innovation 2 , 4 (2021).
[32] Xue, W., Qiu, W., An, B., Rabinovich, Z., Obraztsova, S., and Yeo, C. K.Mis-spoke or mis-lead: Achieving robustness
in multi-agent communicative reinforcement learning. arXiv preprint arXiv:2108.03803 (2021).
[33] Y ang, C., Y ang, G., and Zhang, J.Learning individual difference rewards in multi-agent reinforcement learning. In
Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems (2023), pp. 2418-2420.
[34] Young, H. P. Social norms and economic welfare. European Economic Review 42 , 3-5 (1998), 821-830.
[35] Zhang, J., Xie, N., Zhang, X., and Li, W. An online auction mechanism for cloud computing resource allocation and
pricing based on user evaluation and cost. Future Generation Computer Systems 89 (2018), 286-299.
[36] Zhou, M., Liu, Z., Sui, P., Li, Y., and Chung, Y. Y. Learning implicit credit assignment for cooperative multi-agent
reinforcement learning. Advances in neural information processing systems 33 (2020), 11853-11864.
, Vol. 1, No. 1, Article . Publication date: February 2026.
