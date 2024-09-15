# Awesome LLM Strawberry (OpenAI o1)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![visitor badge](https://visitor-badge.lithub.cc/badge?page_id=hijkzzz.awesome-llm-strawberry&left_text=Visitors) ![GitHub stars](https://img.shields.io/github/stars/hijkzzz/Awesome-LLM-Strawberry?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/hijkzzz/Awesome-LLM-Strawberry?color=9cf) [![GitHub license](https://img.shields.io/github/license/hijkzzz/Awesome-LLM-Strawberry)](https://github.com/hijkzzz/Awesome-LLM-Strawberry/blob/main/LICENSE)

This is a collection of research papers & blogs for **OpenAI Strawberry(o1) and Reasoning**.

And the repository will be continuously updated to track the frontier of LLM Reasoning.

## Blogs

- [OpenAI] [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [OpenAI] [OpenAI o1-mini Advancing cost-efficient reasoning](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning)
- [Nathan Lambert] [OpenAI’s Strawberry, LM self-talk, inference scaling laws, and spending more on inference](https://www.interconnects.ai/p/openai-strawberry-and-inference-scaling-laws)
- [Tibor Blaho] [Summary of what we have learned during AMA hour with the OpenAI o1 team](https://twitter-thread.com/t/1834686946846597281)
  
## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - code
  - experimental environments and datasets
```

### 2024
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
  - Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean Hendryx, Summer Yue, Hugh Zhang
- [An Empirical Analysis of Compute-OptimaInference for Problem-Solving with LanguageModels](https://arxiv.org/abs/2408.00724)
  - Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang
- [Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling](https://www.arxiv.org/abs/2408.16737)
  - Hritik Bansal, Arian Hosseini, Rishabh Agarwal, Vinh Q. Tran, Mehran Kazemi
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
  - Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
  - Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
  - Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
  - Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)
  - Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
  - Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston
- [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271)
  - Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He, Pang Wei Koh, Bryan Hooi
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
  - Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
- [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)
  - Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan et al.
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)
  - Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
  - Guoxin Chen, Minpeng Liao, Chengxi Li, Kai Fan.
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)
  - Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang.
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
  - Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P. Lillicrap, Kenji Kawaguchi, Michael Shieh.
 
### 2023
- [Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training](https://arxiv.org/abs/2309.17179)
  - Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, Jun Wang
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
  - Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
- [Don’t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)
  - Liu, Jiacheng, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz.

### 2021
- [Scalable Online Planning via Reinforcement Learning Fine-Tuning](https://arxiv.org/abs/2109.15316)
  - Arnaud Fickinger, Hengyuan Hu, Brandon Amos, Stuart Russell, Noam Brown.
 
## Evaluation
- [AryanDLuffy] [codeforces](https://codeforces.com/blog/entry/133962)


## Twitter
- <img src="https://github.com/user-attachments/assets/4670514c-e6fa-474f-abea-c3f6ad01e41a" width="300px">
- <img src="https://github.com/user-attachments/assets/b390ccea-9773-4a96-ba02-40d917473402" width="300px">

