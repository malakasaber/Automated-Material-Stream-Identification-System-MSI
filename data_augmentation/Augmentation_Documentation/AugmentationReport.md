0\. My Role

Data Prep + Augmentation + Balancing

Tasks:

\- Load dataset, organize folders, inspect quality

\- Apply augmentation (rotate, flip, scale, brightness, noise, cropping…)

\- Ensure each class ≈ same size (500 images each)

\- Save augmented dataset to new directory

\- Document augmentation choices

\- class balancing verification script

Deliverables:

\- Augmentation script (augmentation.py)

\- Balanced dataset report (counts)

NOTE: 

\- Target exactly 500 images/class after augmentation (30%+ increase). Use Albumentations library for robust transforms. Include validation split (80/20) in saved dataset.

\- Add image normalization (mean/std) to preprocessing and save it.



1\. Raw Dataset Class Count:

cardboard -> 259

glass -> 401

metal -> 328

paper -> 476

plastic -> 386

trash -> 110



Total size = 1960 



2\. What should we increase

\- I am asked to increase training sample by minimum 30%

\- increasing each type by 30%:

cardboard 259 -> 336

glass 401 -> 521

metal 328 -> 426

paper 476 -> 619

plastic 386 -> 502

trash 110 -> 143



Total size after increasing each by 30% = 2547

\- The desired increase 30% of the original 1960 is 2548, 2547 is the increase

\- But, the classes are completely imbalanced

we can increase all to the maximum 619 but this adds too much images to small categories

so I choose to only keep 500 per category as it is like a middle point between the big categories and the small categories (this approach adds more than 30% increase to some classes and less than 30% increase to the other big classes paper and glass BUT, it adds more than 30% on the total dataset)  

with 500 total size becomes = 3000 which is 53.5% increase on the dataset



3\. Finally we need to increase on each category:

cardboard increase 241

glass increase 99 

metal increase 172

paper increase 24

plastic increase 114

trash increase 390



4\. What Augmentation Techniques for each category to ensure logical generated pictures

|   Class        |  Curr   | Target | increase |                    Recommended Augmentation                               |

| ---------------| --------| ------ | ---------| ------------------------------------------------------------------------- |

| cardboard      | 259     | 500    | +241     | Rotate, Horizontal Flip, Brightness, Slight Noise, Crop                   |

| glass          | 401     | 500    | +99      | Rotate, Horizontal Flip, Brightness, Blur, Slight Noise                   |

| metal          | 328     | 500    | +172     | Rotate, Flip, Scale, Contrast, Noise, Crop                                |

| paper          | 476     | 500    | +24      | Minor Rotate, Flip, Brightness, Minor Noise                               |

| plastic        | 386     | 500    | +114     | Rotate, Flip, Brightness/Contrast, Noise, Crop                            |

| trash          | 110     | 500    | +390     | **Heavy augmentation**: Rotate, Flip, Scale, Brightness/Contrast, Noise, Crop |



Considerations I took:

* For large increases (like trash), used combinations of transformations to get more diversity.
* For small increases (like paper), used subtle transformations to avoid creating unrealistic images.
* Avoided augmentations that distort class identity like rotating cardboard upside-down.
