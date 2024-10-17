  # Awesome LLM Strawberry (OpenAI o1)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![visitor badge](https://visitor-badge.lithub.cc/badge?page_id=hijkzzz.awesome-llm-strawberry&left_text=Visitors) ![GitHub stars](https://img.shields.io/github/stars/hijkzzz/Awesome-LLM-Strawberry?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/hijkzzz/Awesome-LLM-Strawberry?color=9cf) [![GitHub license](https://img.shields.io/github/license/hijkzzz/Awesome-LLM-Strawberry)](https://github.com/hijkzzz/Awesome-LLM-Strawberry/blob/main/LICENSE)

This is a collection of research papers & blogs for **OpenAI Strawberry(o1) and Reasoning**.

And the repository will be continuously updated to track the frontier of LLM Reasoning.

## OpenAI Docs
- [https://platform.openai.com/docs/guides/reasoning](https://platform.openai.com/docs/guides/reasoning)
- <img src="https://github.com/user-attachments/assets/b165cb20-9202-4951-8783-6b2f7e0d6071" width="600px">


## Blogs

- [OpenAI] [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [OpenAI] [OpenAI o1-mini Advancing cost-efficient reasoning](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning)
- [OpenAI] [Finding GPT-4’s mistakes with GPT-4](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)
- [Tibor Blaho] [Summary of what we have learned during AMA hour with the OpenAI o1 team](https://twitter-thread.com/t/1834686946846597281)
- [Nathan Lambert] [OpenAI’s Strawberry, LM self-talk, inference scaling laws, and spending more on inference](https://www.interconnects.ai/p/openai-strawberry-and-inference-scaling-laws)
- [Nathan Lambert] [Reverse engineering OpenAI’s o1](https://www.interconnects.ai/p/reverse-engineering-openai-o1)
- [Andreas Stuhlmüller, jungofthewon] [Supervise Process, not Outcomes](https://www.alignmentforum.org/posts/pYcFPMBtQveAjcSfH/supervise-process-not-outcomes)
- [Nouha Dziri] [Have o1 Models Cracked Human Reasoning?](https://substack.com/home/post/p-148782195)
- [Wei Shen] [Generalization Progress in RLHF: Insights into the Impact of Reward Models and PPO](https://difficult-link-dd7.notion.site/4e0cbb325aaf458da710f0b36dbb239c?v=c9231e8c988b4d66a1d2dc34df4cf7b5)
- [Rishabh Agarwal] [Improving LLM Reasoning using Self-generated data: RL and Verifiers](https://rosanneliu.com/dlctfs/dlct_240531.pdf)
- [Dominater069] [Codeforces - Analyzing how good O1-Mini actually is](https://codeforces.com/blog/entry/133887)

## Talks
- [Noam Brown] [Parables on the Power of Planning in AI: From Poker to Diplomacy](https://www.youtube.com/watch?app=desktop&v=eaAonE58sLU)
- [Noam Brown] [OpenAI o1 and Teaching LLMs to Reason Better](https://www.youtube.com/watch?v=jPluSXJpdrA&t=1669s)
- [Hyung Won Chung] [Don't teach. Incentivize.](https://www.youtube.com/watch?v=kYWUEV_e2ss)

## Twitter
- [OpenAI Developers] [We’re hosting an AMA for developers from 10–11 AM PT today.](https://x.com/OpenAIDevs/status/1834608585151594537)
- <img src="https://github.com/user-attachments/assets/4670514c-e6fa-474f-abea-c3f6ad01e41a" width="300px">
- <img src="https://github.com/user-attachments/assets/b390ccea-9773-4a96-ba02-40d917473402" width="300px">
- <img src="https://github.com/user-attachments/assets/aa0678fa-28eb-4b2a-a6ff-c0d123568f22" width="300px">
- <img src="https://github.com/user-attachments/assets/88896f70-017d-4520-ac56-370a023cfe45" width="300px">
- <img src="https://github.com/user-attachments/assets/fbbf78e4-d34c-4b7b-8163-f8c7288f56a6" width="300px">
- <img src="https://github.com/user-attachments/assets/cb1cc1e6-35d4-4567-891a-4e5aca8fa175" width="300px">
- <img src="https://github.com/user-attachments/assets/d3fd109b-0c97-4a94-931e-919b3b2f75f4" width="300px">


## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - code
  - experimental environments and datasets
```

### Relevant Paper from OpenAI o1 [contributors](https://openai.com/openai-o1-contributions/)
- [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095)
  - Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, Aleksander Mądry
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
  - Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
- [Generative Language Modeling for Automated Theorem Proving](https://arxiv.org/abs/2009.03393)
  - Stanislas Polu, Ilya Sutskever
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
  - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
  - Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- [LLM Critics Help Catch LLM Bugs](https://arxiv.org/abs/2407.00215)
  - Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trebacz, Jan Leike
- [Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802)
  - William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, Jan Leike
- [Scalable Online Planning via Reinforcement Learning Fine-Tuning](https://arxiv.org/abs/2109.15316)
  - Arnaud Fickinger, Hengyuan Hu, Brandon Amos, Stuart Russell, Noam Brown.
- [Improving Policies via Search in Cooperative Partially Observable Games](https://arxiv.org/abs/1912.02318)
  - Adam Lerer, Hengyuan Hu, Jakob Foerster, Noam Brown.

### 2024
- [An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models](https://arxiv.org/abs/2408.00724)
  - Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang
- [Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling](https://www.arxiv.org/abs/2408.16737)
  - Hritik Bansal, Arian Hosseini, Rishabh Agarwal, Vinh Q. Tran, Mehran Kazemi
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
  - Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
  - Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini
- [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)
  - Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, Aleksandra Faust
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
  - Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
  - https://github.com/ezelikman/quiet-star
- [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)
  - Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
  - Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal
- [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146)
  - Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
  - Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, Abhinav Rastogi
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)
  - Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, Zhifang Sui
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
  - Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean Hendryx, Summer Yue, Hugh Zhang
- [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199)
  - Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, Rafael Rafailov
- [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271)
  - Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He, Pang Wei Koh, Bryan Hooi
- [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)
  - Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan et al.
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)
  - Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
  - Guoxin Chen, Minpeng Liao, Chengxi Li, Kai Fan.
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)
  - Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang.
- [MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time](https://arxiv.org/abs/2405.16265)
  - Jikun Kang, Xin Zhe Li, Xi Chen, Amirreza Kazemi, Qianyi Sun, Boxing Chen, Dong Li, Xu He, Quan He, Feng Wen, Jianye Hao, Jun Yao.
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
  - Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P. Lillicrap, Kenji Kawaguchi, Michael Shieh.
- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875)
  - Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma.
- [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/abs/2409.12183)
  - Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, Greg Durrett
- [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)
  - Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, Sebastian Riedel
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/pdf/2402.10200)
  - Xuezhi Wang, Denny Zhou
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
  - Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- [Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2406.09136)
  - Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, Min Lin
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)
  - Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, Hang Li
- [VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment](https://arxiv.org/abs/2410.01679)
  - Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, Nicolas Le Roux
- [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229)
  - Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar
- [Evaluation of OpenAI o1: Opportunities and Challenges of AGI](https://arxiv.org/abs/2409.18486)
  - Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, Chao Cao, Hanqi Jiang, Hanxu Chen, Yiwei Li, Junhao Chen, etc.
- [Evaluating LLMs at Detecting Errors in LLM Responses](https://arxiv.org/abs/2404.03602)
  - Ryo Kamoi, Sarkar Snigdha Sarathi Das, Renze Lou, Jihyun Janice Ahn, Yilun Zhao, Xiaoxin Lu, Nan Zhang, Yusen Zhang, Ranran Haoran Zhang, Sujeeth Reddy Vummanthala, Salika Dave, Shaobo Qin, Arman Cohan, Wenpeng Yin, Rui Zhang
- [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)
  - Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, Bo An
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)
  - Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang
  
### 2023
- [Training Chain-of-Thought via Latent-Variable Inference](https://arxiv.org/pdf/2312.02179)
  - Du Phan, Matthew D. Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous
- [Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training](https://arxiv.org/abs/2309.17179)
  - Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, Jun Wang
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
  - Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
- [Don’t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)
  - Liu, Jiacheng, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz.
- [Certified reasoning with language models](https://arxiv.org/pdf/2306.04031)
  - Gabriel Poesia, Kanishk Gandhi, Eric Zelikman, Noah D. Goodman
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
  - Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou

### 2022
- [Chain of Thought Imitation with Procedure Cloning](https://arxiv.org/abs/2205.10816)
  - Mengjiao Yang, Dale Schuurmans, Pieter Abbeel, Ofir Nachum.
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
  - Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- [Solving math word problems with processand outcome-based feedback](https://arxiv.org/abs/2211.14275)
  - Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins

### 2021
- [Scaling Scaling Laws with Board Games](http://arxiv.org/abs/2104.03113)
  - Andy L. Jones.
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/pdf/2112.00114)
  - Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus Odena

### 2017
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815v1)
  - David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis.

## Projects
- [openreasoner] [OpenR](https://openreasoner.github.io/)
- [OpenO1 Team] [Open-Source O1](https://opensource-o1.github.io/)
- [GAIR-NLP] [O1 Replication Journey: A Strategic Progress Report](https://github.com/GAIR-NLP/O1-Journey)
- [Maitrix.org] [LLM Reasoners](https://github.com/maitrix-org/llm-reasoners)
- [bklieger-groq] [g1: Using Llama-3.1 70b on Groq to create o1-like reasoning chains](https://github.com/bklieger-groq/g1)
- [o1-chain-of-thought] [Transcription of o1 Reasoning Traces from OpenAI blog post](https://github.com/bradhilton/o1-chain-of-thought)
- [toyberry] [Toyberry: A end to end tiny implementation of OpenAI's o1 reasoning system using MCTS and LLM as backend](https://github.com/ack-sec/toyberry)
