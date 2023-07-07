---
layout: post
title: "Top Machine Learning in Cybersecurity Trends to watch in 2022"
author: Maria Zorkaltseva
categories: [Machine learning, Cybersecurity]
tags: 
    - machine learning
    - fraud
    - adversarial attacks
feature-img: "assets/img/sample_feature_img.png"
excerpt_begin_separator: <!--excerpt-->
excerpt_separator: <!--more-->
comments: true
date:	2022-06-20
---

<!--excerpt-->

  According to [Open Web Application Security Project (OWASP)](https://owasp.org/) ratings of top cybersecurity risks, among which [web application risks](https://owasp.org/www-project-top-ten/), [mobile](https://owasp.org/www-project-mobile-top-10/) and [IoT devices risks](https://owasp.org/www-project-internet-of-things/), [machine learning systems](https://owasp.org/www-project-machine-learning-security-top-10/) vulnerabilities, the problem of protection in the digital space continues to be extremely relevant. Here we will consider actual challenges which are arising from the conjunction of machine learning and cybersecurity applications in different areas such as IoT ecosystems, targeted advanced persistent threats, fraud and anomalies detection, etc. Also, you will find interesting ideas for your future machine learning projects.
  
![](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/0*bVhDjwB4IAuCGp_1)

<!--more-->

### Internet of Things (IoT) systems

Due to the computational resources constraints of IoT devices and its insecure authorization/network protocols, [the problem of its protection](https://www.intellectsoft.net/blog/biggest-iot-security-issues/) is currently acute. Different layers of IoT ecosystem should be protected: Perception Layer, Network Layer and Application Layer. Therefore, researchers are considering using machine learning algorithms to monitor network traffic; defend against botneck attacks, ransomwares, crypto-jacking, advanced persistent threats (APT), distributed-denial-of-service (DDoS) and man-in-the-middle (MITM) attacks; provide IoT applications security, etc.

![IoT Architecture](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/1*3dJYba4cj7OvidJHJtsr6A.png)

*IoT Architecture, source: “[A Survey of Machine and Deep Learning Methods for Internet of Things (IoT) Security](https://arxiv.org/pdf/1807.11023.pdf)”*

Despite the success of ML algorithms application in other fields, in the field of IoT layers security, there are still some challenges and problems to be solved. One such problem is the **lack of data** or lack of **high-quality data**. For example, to evaluate ML approach to protect from botnet attacks, researchers are attempting to get data by infecting IoT devices with widely known Mirai and BASHLITE like it shown in “[N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders](https://arxiv.org/pdf/1805.03409.pdf)”. [Some authors](https://www.atlantis-press.com/journals/ijcis/25905181) have made an attempt to simulate artificial data by using Cooja IoT simulator, while [others ](https://ieeexplore.ieee.org/document/9728969)have used testbeds to simulate different kinds of attacks.

Another one challenge is related to computational resources and energy efficiency constraints of IoT devices that can be considered as crucial bottlenecks for deep learning models deployment and usage at the edge. However, there are some techniques to reduce model size and complexity, such as [quantization](https://arxiv.org/pdf/2103.13630.pdf), [pruning](https://arxiv.org/abs/1803.03635), [knowledge distillation](https://arxiv.org/pdf/1910.08381.pdf), network architecture search. Also, IoT environment is dynamic and it’s very important to develop a continuous training pipeline.

Let us imagine that we have successfully deployed ML model within IoT ecosystem. Here, the problem of data and ML model security is appearing. Recent advances in ML algorithms have enabled them to be used in breaking cryptographic implementations, for example, RNN model can learn patterns to break Enigma machine decryption function. Another problem is the data (user data, datasets, ML model artifacts) leakage/privacy, if attacker would know structure of the data/ML model and would have access to it, he will have a possibility for data/model poisoning, ML model fooling by generating adversarial samples and etc.

**Open datasets:**

* [The Bot IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)
* [N-BaIoT Dataset to Detect IoT Botnet Attacks](https://www.kaggle.com/mkashifn/nbaiot-dataset)
* [IOT Botnets Attack Detection Dataset](https://www.kaggle.com/saurabhshahane/anomaly-detection-using-deep-learning)
* [DDoS Botnet Attack](https://www.kaggle.com/datasets/siddharthm1698/ddos-botnet-attack-on-iot-devices)
* [Edge-IIoT Set](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
* [UNSW-NB 15 dataset](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
* [MQTT-set](https://www.kaggle.com/datasets/cnrieiit/mqttset)

**ML projects ideas:**

* generation/augmention existing IoT data using General Adversarial Networks (GAN) or any other generative model;
* noise reduction in IoT data using Denoising Autoencoder (DAE) models;
* network traffic anomalies detection/time series forecasting;
* IoT benign/malware applications classification and categorization;
* experimenting with quantization/pruning/knowledge distillation to optimize model for deployment purpose.

### Advanced Persistent Threats (APT)

An advanced persistent threat (APT) can be defined as a targeted cyber attack which can bypass all kinds of security software (firewalls, antiviruses) and remain invisible for quite a long period of time. Among the well-known examples of such attacks is [**Stuxnet**](https://en.wikipedia.org/wiki/Stuxnet#:~:text=Stuxnet%20is%20a%20malicious%20computer,the%20nuclear%20program%20of%20Iran.), which targeted SCADA (Supervisory Control and Data Acquisition) systems to cause substantial damage in the nuclear program of Iran. [**Epic Turla**](https://www.kaspersky.com/resource-center/threats/epic-turla-snake-malware-attacks), which was identified by Kaspersky, aimed to infect the systems of government agencies, state departments, military agencies and embassies in more than 40 countries worldwide. [**Deep panda**](https://teampassword.com/blog/who-is-deep-panda-and-how-can-you-protect-yourself) which was an attack carried out to obtain the staff information of the US Intelligence Service, and was probably of Chinese origin.

Techniques which are used to bypass security are exploitation of well-known vulnerabilities, malware usage, spear phishing, zero day vulnerability, watering hole attack, social engineering. A good practice to protect against such kind of attacks would be joint usage of ELK (Elasticsearch, Logstash, Kibana) stack to monitor any system anomalies and ML techniques. Due to the time-stretched and complex nature of APT attacks, in most cases it is not enough to use only machine learning protection module; such a model works in conjunction with other protection methods like signature-based methods.

As time-spread APT nature, many researchers in this field try to build solutions based on recurrent neural network (RNN) models which are aimed to process sequences. For instance, in [“Advance persistent threat detection using long short term memory (LSTM) neural networks”](https://link.springer.com/chapter/10.1007/978-981-13-8300-7_5) paper, authors used LSTM model that takes Splunk SIEM event logs as input to detect APT espionage. Another approach is to construct multi-module systems like in “[Detection of advanced persistent threat using machine-learning correlation analysis](https://www.sciencedirect.com/science/article/pii/S0167739X18307532)” to detect multi-stage APT malwares by using ML and correlation analysis. The novelty of this research lies in the detection of APT across all life cycle phases. In “[A context-based detection framework for advanced persistent threats](https://ieeexplore.ieee.org/abstract/document/6542528)” authors used events from different sources, i.e. VPN logs, firewall logs, IDS logs, authentication logs, system event logs which were passed as data source to the detection engine. From these logs, the context of attack is identified using correlation rules. The suspicious activities are identified by matching the attack contexts using a signature database. As you can see, researchers approach the problem from different angles, and there is still plenty of room for creativity.

**Open datasets:**

* [Dataset for a Apt Identification Triage System](https://github.com/GiuseppeLaurenza/I_F_Identifier)
* [APT Malware Dataset Containing over 3,500 State-Sponsored Malware Samples](https://github.com/cyber-research/APTMalware)
* [APT-EXE, APT-DLL, APT-IoC](https://github.com/aptresearch/datasets)
* [NSL-KDD dataset](https://github.com/jmnwong/NSL-KDD-Dataset)
* [Advanced Persistent Threat (APT) Malware Dataset — 2020](https://cybersciencelab.com/advanced-persistent-threat-apt-malware-dataset/)

**ML projects ideas:**

* APT data simulation/generation;
* development of Intrusion Detection System (IDS) for APT detection;
* classification of APT malwares/normal malwares;
* network traffic clusterization to detect hidden patterns.

### Fraud detection

The concept of fraud in the light of information security is quite extensive and affects various areas such as financial institutions, retail, logistic organizations, insurance companies, gaming/gambling, healthcare sector, social communities, governances and etc. Fraudsters try to get personal data, for which they build multi-vector attacks that may include: social engineering methods (e.g., spear phishing), malwares, account takeover scams, impersonation fraud. Recent examples of multi-vector fraud attacks include cyber attacks using the SWIFT-related banking infrastructure, ATM infections, remote banking systems and point-of-sale (POS) terminal networks, making changes in Payment Service Provider (PSP) databases to “play” with account balances, as well as the so-called [supply-chain attacks](https://en.wikipedia.org/wiki/Supply_chain_attack). Digitization and automation of customer experience not only improve good customers outcomes, but also open the doors for fraudulent activities automatization (bot attacks): spam registrations, automation of logins for account takeover, automated testing stolen credit card credentials.

In case of supervised machine learning (kNN, logistic regression, SVM, random forests and gradient boosting, feed-forward neural networks, recurrent and convolutional neural networks, SOTA solutions which combine different deep learning models, e.g. autoencoder plus LSTM), the problem of fraud detection is usually considered as a binary classification problem of fraudulent/legitimate data samples. However, labeled data is not always available, and researchers resort to unsupervised machine learning to solve [anomaly/novelty detection](https://github.com/yzhao062/pyod) (Local Outlier Factor, iForest, One-Class SVM, One-class GAN, Variational AutoEncoder).

Challenges which are rising in fraud detection area using ML:

* **Unbalanced data.** Fraudulent samples are rare in comparison to legitimate traffic, two naive methods to overcome this problem is to use undersampling/oversampling technique, another one is Synthetic Minority Oversampling Technique strategy (SMOTE). It’s also possible to use generative models to sample new points from probabilistic distribution of fraud samples.
* **Data shift.** Fraudsters behaviour is dynamic, and evolve over time, as well as customers behaviour patterns. Usually training ML models on train set, we assume that it is valid on the test set (if we did everything right), but in case of data drift the hypothesis will be broken.
* **Fairness of ML model is highly important.** Developed ML model should treat customers diversity equally, that is before deployment into production, one should evaluate models across different customer groups. You can use TFX (TensorFlow Extended) framework which is good for all stages of driven ML model into production. This framework provides service for [fairness estimation](https://www.tensorflow.org/tfx/guide/fairness_indicators).
* **High presence of categorical variables.** Most learning algorithms can not handle categorical variables with a large number of levels directly, that is why optimal feature encoding strategy should be used to minimize RAM usage. Different supervised (generalized linear mixed model encoder, James-Stein estimator, target encoding, quantile encoder) and unsupervised (hashing, one-hot, count, Helmet coding) methods for [categorical features encoding](https://github.com/scikit-learn-contrib/category_encoders) can be used.
* **Non-linear anomalies behaviour.** After model was deployed into production, the ML engineer work is not finished yet. It’s necessary to continuously monitor data, for example, you can use Kibana dashboards from ELK stack. There should be chosen optimal time frame length and schedule to retrain your model on new upcoming data to prevent data/concept drift. For instance, model retraining can happen automatically according to schedule using AirFlow framework.
* **Trade-off between fraudsters stopping and making friction for legitimate users.** There should be optimal trade-off threshold which will minimize False Positives (legitimate user detected as fraudster) and maximize False Negatives (fraudster detected as legitimate user) triggerings of ML model.

**Open datasets:**

* [Credit card fraud detection Challenge](https://www.kaggle.com/mlg-ulb/creditcardfraud/home)
* [German Credit card Fraud Data](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)
* [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv)
* [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
* [Digital Advertising Fraud](https://www.kaggle.com/datasets/anmolkumar/digital-advertising-conversion-fraud?select=readme.txt)
* [Job Fraud Dataset](https://github.com/k-means/job_fraud_detection)

**ML projects ideas:**

* experiments with different generative models for fraud samples generation;
* classification of fraudulent/legitimate activities;
* use Graph neural networks to create Anti-Money Laundering model;
* phishing URL detection;
* detection of fake accounts in social networks.

### Targeted ransomwares

Malware is harmful software that comes in various forms such as viruses, worms, rootkits, spyware, trojans, ransomware, and so on. Ransomware is a type of malware that encrypts all of a user’s important files and demands a ransom to unlock them. In general, the ransom is requested in digital currency, and the anonymity of digital currency allows attackers to avoid prosecution. It also provides justification for an increase in the amount of ransomware attacks. There are two types of ransomware: **crypto ransomware** and **locker ransomware**. Crypto malware encrypts system files, rendering them inaccessible. File-Locker ransomware is another name for crypto ransomware. Locker ransomware does not corrupt data; instead, it inhibits victims from accessing their systems by displaying a window that never closes or locking their desktop. The functioning of ransomware is similar to that of benign software in that it operates without anybody being aware of it. As a result, detection of ransomware in zero-day attacks is critical at this time.

![](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/1*BBmJvc0rB1S75QRAPrMNLw.png)

*Cyber kill chain based taxonomy diagram of the ransomware features, source: [“A Cyber-Kill-Chain based taxonomy of crypto-ransomware features”](https://link.springer.com/article/10.1007/s11416-019-00338-7)*

There are several approaches and procedures for detecting ransomware. **Static analysis** based methods disassemble source code without executing it. However, they have a significant false positive rate and are incapable of detecting obfuscated ransomware. To combat these challenges, researchers are turning to **dynamic behavior analysis-based** tools that monitor the interactions of the executed code in a virtual environment and extract executed API sequences. However, these detection methods are slower and demand a lot of memory. Machine learning is ideally suited for analyzing the behavior of any process or application. Researchers proposed two different ways to detect ransomware using machine learning: **host-based** and **network-based**. In case of host-based analyzing approach file system activity, API calls, registry key operations, energy consumption patterns and other features are monitored. In network-based approach, malicious and benign traffic samples/logs are analyzed; for example, destination and source IP address, protocol type, source and destination port number, the total number of bytes and packets per conversation. Also, protection solutions can be divided into early-stage ransomware detection (before encryption) and post encryption detection, which are not such relevant.

The advantage of the ML-based protection approach over signature-based methods is that it is able to detect zero-day** **attacks using anomaly detection methods. In [“Zero-day aware decision fusion-based model for crypto-ransomware early detection”](https://publisher.uthm.edu.my/ojs/index.php/ijie/article/view/2828/1765) authors used group of one-class SVM models which were trained on benign programs. The results of this classifier were integrated using the majority voting method. In [“Automated Analysis Approach for the Detection of High Survivable Ransomware”](http://itiis.org/digital-library/23567) researchers proposed a framework for the behavioral-based dynamic analysis of high survivable ransomware (HSR) with integrated valuable feature sets. Term Frequency-Inverse document frequency (TF-IDF) was employed to select the most useful features from the analyzed samples. Support Vector Machine (SVM) and Artificial Neural Network (ANN) were utilized to develop and implement a machine learning-based detection model able to recognize certain behavioral traits of high survivable ransomware attacks.

The main challenges regarding ransomware area:

* irrelevant and redundant system calls, obfuscation/ packing technique usage to bypass ML model detection can be used;
* the ransomware detection systems are platform-dependent;
* diversity of ransomware families;
* datasets used to train are synthetic and extracted from specific sources, i.e., pseudo-real world events;
* not all the detection studies available in the literature are practical to implement.

**Open datasets:**

* [Ransomware Dataset RISSP group](http://rissgroup.org/ransomware-dataset/)
* [Ransomware bitcoin datasets](https://github.com/behas/ransomware-dataset)
* [ISOT Ransomware dataset](https://www.uvic.ca/engineering/ece/isot/datasets/botnet-ransomware/index.php)
* [Microsoft Malware Classification Challenge (BIG 2015)](https://www.kaggle.com/c/malware-classification)

**ML projects ideas:**

* host-based ransomware detection using static/dynamic/hybrid features;
* ransomware by network traffic analysis, network traffic clusterization;
* ransomware/normal malware/benign applications classification;
* ransomware family classification.

### Adversarial attacks and Machine Learning Apps security

Traditional antivirus solutions are only effective half of the time, it use signature-based methods and heuristics to search through already seen attacks. Attackers have learned to overcome that protection using polymorphic malwares and obfuscation techniques. Next-generation antivirus (NGAV) programs hit the market, NGAV uses artificial intelligence, behavioural patters and predictive modeling techniques to identify malware and malicious behavior in advance. Thus, adversarial machine learning attacks enter the scene, here is the math formulation of the problem:

![](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/1*slTOGqoS1agZH9AX6be6kA.png)

Minimization problem for adversarial examples generationThe input sample *x*, correctly classified by the classifier *f*, is perturbed with *r* such that the resulting adversarial example, *x* + *r*, remains in the input domain *D* but is assigned a different label than *x*.

As can be seen from the problem formalization, such attacks are easy to carry out in the field of computer vision, one can simply add noise to the input image in the direction of corresponding gradient, so that it will be misclassified. This fact is well illustrated by fast gradient sign method.

![](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/1*ZJP0HTk62ubPWE07qi1YVQ.png)

*Fast gradient sign method, source: [https://www.tensorflow.org/tutorials/generative/adversarial\_fgsm](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)*

![](/assets/img/2022-06-20-top-machine-learning-in-cybersecurity/1*-KDbP3R0YLFhxFoRUCP8tw.png)

In the computer vision domain, the adversary can change every pixel’s colour (to a different valid colour) without creating an “invalid picture” as part of the attack. However, in the cyber security domain, modifying an API call (e.g. replacing WriteFile() call with ReadFile() ) might cause executable to perform differently or even crash. Network packages perturbation in order to evade network intrusion detection module is also challenging task. Small changes not perceived by human eye are not possible in cybersecurity domain. However, some transparency in the process of machine learning makes it easier for hackers to make attacks against ML modules. An example would be the use of transfer learning, where ML engineers utilize common pre-trained model architectures or [phishing URL detection](https://mariazork.github.io/machine%20learning/2021/07/29/phishing-url-detection.html) task where common NLP features are employed. In that case, it would be gray-box attack, when hacker has partial information about ML model training process. [This review](https://arxiv.org/abs/2007.02407) could be a good starting point to explore adversarial attacks in various cybersecurity areas, its types and purposes, as well as defense methods.

**ML projects ideas:**

* generate adversarial examples using GAN and test your ML model robustness using these examples;
* deepfake detection model like MesoNet/ MesoInception.

#### Further reading:

Internet of Things (IoT) systems:

* [A Survey of Machine and Deep Learning Methods for Internet of Things (IoT) Security](https://arxiv.org/pdf/1807.11023.pdf)
* [N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders](https://arxiv.org/pdf/1805.03409.pdf)
* [Deep Learning for Detection of Routing Attacks in the Internet of Thing](https://www.atlantis-press.com/journals/ijcis/25905181)
* [OWASP IoT Top 10 based Attack Dataset for Machine Learning](https://ieeexplore.ieee.org/document/9728969)
* [Detecting Malware in Cyberphysical Systems Using Machine Learning: a Survey](https://www.koreascience.or.kr/article/JAKO202116739374210.page)

Advanced Persistent Threat (APT):

* [A New Proposal on the Advanced Persistent Threat: A Survey](https://www.mdpi.com/2076-3417/10/11/3874)
* [A Survey of Machine Learning Techniques Used to Combat Against the Advanced Persistent Threat](https://link.springer.com/chapter/10.1007/978-981-15-0871-4_12)
* [DMAPT: Study of Data Mining and Machine Learning Techniques in Advanced Persistent Threat Attribution and Detection](https://www.intechopen.com/chapters/77974)
* [Detection of advanced persistent threat using machine-learning correlation analysis](https://www.sciencedirect.com/science/article/pii/S0167739X18307532)

Fraud detection:

* [Types of fraud](https://www.aura.com/learn/examples-of-fraud)
* [Deep Learning for Anomaly Detection: A Review](https://dl.acm.org/doi/10.1145/3439950)
* [Credit card fraud detection using machine learning: A survey](https://arxiv.org/pdf/2010.06479.pdf)

Targeted ransomwares:

* [A Survey on Machine Learning-Based Ransomware Detection](https://link.springer.com/chapter/10.1007/978-981-16-6890-6_13)
* [Ransomware Detection Using the Dynamic Analysis and Machine Learning: A Survey and Research Directions](https://www.mdpi.com/2076-3417/12/1/172/htm)
* [A Survey on Detection Techniques for Cryptographic Ransomware](http://dataset.tlm.unavarra.es/ransomware/articles/IEEEAccess.pdf)
* [Automated Analysis Approach for the Detection of High Survivable Ransomware](http://itiis.org/digital-library/23567)

Adversarial attacks and Machine Learning Apps security:

* [Adversarial Machine Learning Attacks and Defense Methods in the Cyber Security Domain](https://arxiv.org/abs/2007.02407)
* [Adversarial Attacks and Defenses in Deep Learning](https://www.sciencedirect.com/science/article/pii/S209580991930503X?via%3Dihub)
* [Adversarial Attacks on Deep Learning Models in Natural Language Processing: A Survey](https://arxiv.org/abs/1901.06796)
* [Black-box Adversarial Attacks Against Deep Learning Based Malware Binaries Detection with GAN](https://ecai2020.eu/papers/1118_paper.pdf)
* [AT-GAN: A Generative Attack Model for Adversarial Transferring on Generative Adversarial Net](https://www.researchgate.net/publication/332463200_AT-GAN_A_Generative_Attack_Model_for_Adversarial_Transferring_on_Generative_Adversarial_Nets?enrichId=rgreq-98304190d497530361ab23744e043a75-XXX&enrichSource=Y292ZXJQYWdlOzMzMjQ2MzIwMDtBUzo4MzUxMjUzMzE4ODYwODBAMTU3NjEyMDgwMzE2OA%3D%3D&el=1_x_3&_esc=publicationCoverPdf)
  