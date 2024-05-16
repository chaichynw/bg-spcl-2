# Brain-Guided Self-Paced Curriculum Learning for Adaptive Human-Machine Interfaces

## Abstract 

![framework.pdf](https://github.com/chaichynw/bg-spcl/files/15329501/framework.pdf)


Human-machine interfaces (HMIs) face several challenges. 

1. First, they face overfitting due to limited calibration data on individual users.
2. Second, they experience data distribution shifts owing to changes in user states over time.
3. Third, they experience disruptions from outlier samples caused by user distractions.
   
As a result, there is an increasing demand for **adaptive HMIs capable of adjusting the interface based on the user's state**, even in scenarios with limited data, to address these challenges.
To address this need, we propose a novel framework, **brain-guided self-paced curriculum learning (BG-SPCL)**, for simultaneously considering intentional knowledge from the model and the user's state knowledge from the brain. In particular, our BG-SPCL constrains the curriculum region by filtering samples that disrupt the interface's learning patterns. Subsequently, it determines the difficulty of samples based on the interface's current level, enabling a gradual progression of learning from easy to difficult samples.

## Datasets

We have chosen three public MI datasets to solve and demonstrate common issues in HMIs.

1. **BNCI2014004**
   - This is a representative 2-class motor imagery dataset.

2. **BNCI2015001**
   - We conducted additional tests on this dataset to examine if the adaptive learning algorithm can be applied individually.
   - Previously, such participants were considered to have BCI illiteracy. However, we have established that appropriate learning methodologies can significantly enhance their performance.

3. **Zhou2016**
   - Lastly, our BG-SPCL proves to effectively respond to brain signals, encompassing user dynamics collected over several weeks to months, as demonstrated through the Zhou2016 dataset.
  
The datasets can be downloaded through [Moabb](https://moabb.neurotechx.com/docs/datasets.html). The downloaded data should be added to the 'bg-spcl/data/' folder.


## Evaluation

![evaluation_scheme.pdf](https://github.com/chaichynw/bg-spcl/files/15329498/evaluation_scheme.pdf)

We conducted performance evaluations by dividing them into offline and online modes.

In **offline mode**, we evaluate the generalization ability of the pre-trained model through inter-session performance.
    
    main.py --config_name 'bnci2014004_config' --target_subject 0 --is_test False --online_update False


In **online mode**, we observe how the pre-trained model adapts when unlabeled EEG signals are streamed, utilizing the Zhou2016 dataset, which has three sessions with the **same paradigm**, to understand the form of  model adaptation over a long period.

    main.py --config_name 'bnci2014004_config' --target_subject 0 --is_test True --online_update True

Please note, as indicated in prior research, performance can fluctuate due to user adaptations over multiple sessions. Thus, to exclusively assess the impact of the learning algorithm, we inverted the sequence of training and testing.

