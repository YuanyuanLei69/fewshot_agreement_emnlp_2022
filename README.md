# Read Me

<br/>

**Paper:** Few-Shot (Dis)Agreement Identification in Online Discussions with Regularized and Augmented Meta-Learning<br/>
**Accepted:** The 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Affiliation:** Department of Computer Science and Engineering, Texas A&M University, College Station, Texas, USA

<br/>

## Task Description
In online discussion forums and social media platforms, people express different opinions towards a common topic, by posting their comments or replying to another user's previous comments. We call the comments replying to another comments as *Response* and the comments being replied as *Quote*. Researching the relation between *(Quote, Response)* pairs will enable many opinion mining applications. Specifically, the task is identifying the relation between *(Quote, Response)* pair to be *Agreement*, *Disagreement*, or *Neutral*.

<br/>

## Code Description
* **FewShot.py:** The metric-based meta-learning model (section 3.1) that trains a meta-learner with two key abilities: deriving the attentive class embedding from few provided support examples, and comparing the relation between new instance and each class embedding to make the prediction.
* **FewShot_reg.py:** The meta-learning model with the lexicon-based regularization loss (section 3.2) that makes the meta-learner focus more on the domain-invariant features.
* **FewShot_aug.py:** The meta-learning model with domain-aware task augmentation (section 3.3) that enables the meta-learner to learn domain-specific expressions.
* **FewShot_aug_reg.py:** The meta-learning model with both lexicon-based regularization loss and domain-aware task augmentation, which is also our full model.

<br/>

## Data Description
Our experiments use five datasets, IAC and ABCD are used as training datasets, while AWTP, MaskMandate, and CovidVaccine are used as testing datasets.
* **Internet Argument Corpus (IAC)** dataset [1] annotated post pairs from the website 4forums.com, and also annotated golden domain labels for each pair in a total number of ten domains.
* **Agreement by Create Debaters(ABCD)** dataset [2] collected post pairs from  the website createdebate.com.
* **Agreement in Wikipedia Talk Pages (AWTP)** dataset [3] collected Q-R pairs from LiveJournal Blogs and Wikipedia Edit Discussions.
* **SubReddit-MaskMandate** is our newly annotated dataset, which collected Q-R pairs from a sub forum on reddit.com with the topic of mask mandate. (https://github.com/yuanyuanlei-nlp/SubReddit_agreement_dataset)
* **SubReddit-CovidVaccine** is our newly annotated dataset, which collected Q-R pairs from a sub forum on reddit.com with the topic of COVID vaccine. (https://github.com/yuanyuanlei-nlp/SubReddit_agreement_dataset)

<br/>

## Citation
If you are going to cite this paper, please use the form:

<br/>

## Reference
[1] Marilyn A. Walker, Pranav Anand, Jean E. Fox Tree, Rob Abbott, Joseph King. "A Corpus for Research on Deliberation and Debate." In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC), Istanbul, Turkey, 2012.<br/>
[2] Sara Rosenthal and Kathy McKeown. 2015. I Couldn’t Agree More: The Role of Conversational Structure in Agreement and Disagreement Detection in Online Discussions. In Proceedings of the 16th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 168–177, Prague, Czech Republic. Association for Computational Linguistics.<br/>
[3] Jacob Andreas, Sara Rosenthal, and Kathleen McKeown. 2012. Annotating Agreement and Disagreement in Threaded Discussion. In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12), pages 818–822, Istanbul, Turkey. European Language Resources Association (ELRA).

