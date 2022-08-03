# SVMwithGEEMPC
**This is a supporting material of a research paper (Analyzing large-scale Data Cubes with user-defined algorithms: A cloud-native approach).**
**Please refer to https://doi.org/10.1016/j.jag.2022.102784**

**The project contians scripts to perform a SVM-based land cover classification based on all bands of Sentinel2-L2A.**


**NOTE: The script is only used for testing the performance of platforms, the accuray of classification is not verified.**

1. SVMwithGEE.js is the script to experiment with GEE. (run the script on https://code.earthengine.google.com/)
2. SVMwithMPC.py is the script to experiment with Microsoft Planetary Computer. (run the script on https://planetarycomputer.microsoft.com/)
3. Classification.pkl is the SVC model of Scikit-learn from version 0.23.2.

![image](https://user-images.githubusercontent.com/96739786/147522565-4a3bcd48-0414-4264-8dae-a128b903829f.png)
**Web-based platform for the same test**

![image](https://user-images.githubusercontent.com/96739786/147520436-8e00e0d6-77af-4812-91ab-98cc835637e0.png)
**Result with 36,239 Sentinel2 L2A datasets from 1, July, 2020 to 30, September, 2020 with **Science Earth Platform****

