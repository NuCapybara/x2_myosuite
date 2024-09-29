# x2_myosuite Description
Create a model for x2 exoskeleton with musculoskeletal model (Simulate real-life situation as a human wearing exoskeleton)
- The original musculoskeletal model is created by MyoSuite 
- This repo adapted the lower-limb musculoskeletal model with the exoskeleton, the file path is 
```/x2_myosuite/myosuite/myosuite/simhive/myo_sim/leg/myolegs_compatible.xml```

## The model overview
![image](https://github.com/user-attachments/assets/d8514d6b-5dab-4ae5-b0cd-8c71af02fa14)


## Transparency Mode & Position controller
- We designed two controllers—position control and transparency mode—using Proportional-Derivative (PD) control to regulate joint angles and interaction forces. The transparency mode minimizes exoskeleton resistance, making it feel “transparent” so users can move freely without restriction.

- To investigate the interaction forces between the human and the exoskeleton, we implemented motion controllers for the hips and knees to execute sinusoidal (sin-wave) movements. The attached file includes a simulation animation and provides results on the interaction forces.
```/x2_myosuite/myosuite/myosuite/simhive/myo_sim/leg/transparencyModel_myosuite.py```

