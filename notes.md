# dataset augmentation -> run whole pipeline -> calculate error and loss
# -> generate like 2000 more samples with the box slightly shifted (each face in both directions, with
# many increments) -> we have a dataset with around 100 GT, the augmented 2000 samples, and from that,
# the *2000 box shifts. 

# Then evaluate the final pipeline for all the boxes. Then let AI learn the best box shift, and output
# its corresponding shift parameters. 

# We also need to calculate regret, so compare the objectively best box shift with the AI's output

# So to quickly recap, so we are on the same page:
# 0. Take the ~100 GTs, and generate the augmented samples (~2000) 
# 1. Use traditional CV to detect the box in each scene
# 2. Generate a dataset of box shifts (each face in both directions, with many increments, with every possible combination) for each of the ~2000 augmented samples (=> many million samples)
# 3. Run the whole pipeline for each box shift, and calculate the final error and loss for each box shift
# 4. Train the AI to learn the best box shift, and output its corresponding shift parameters
# 5. Calculate regret (among other metrics), so compare the objectively best box shift with the AI's output