# Data Augmentation using Generative Adversarial Networks

Project for 2019 Regeneron Science Talent Search ([paper](https://drive.google.com/file/d/1-AdguaHoTb9-5ODO-mIdP0eby8zSpOJo/view?usp=sharing))

Main idea: Designed framework to synthesize training samples using Conditional Wasserstein GAN in order to improve accuracy of digit classifier.

Reduced classification error by 10.4% on full MNIST dataset; achieved comparable accuracy (98.0% vs. 98.6%) while only using 1/10 of the original dataset (code implementation: github.com/agongt408/science-fair)

![Framework overview](./images/data-augmentation.png)

## Results

![Synthesized images (full data)](./images/gan-100/samples_0002500.png)
![Synthesized images (full data)](./images/gan-100/samples_0005000.png)
![Synthesized images (full data)](./images/gan-100/samples_0010000.png)

Full data (2500, 5000, 10000 training episodes)

![Synthesized images (10%)](./images/gan-10/samples_0002500.png)
![Synthesized images (10%)](./images/gan-10/samples_0005000.png)
![Synthesized images (10%)](./images/gan-10/samples_0010000.png)

10% data (2500, 5000, 10000 training episodes)

![Synthesized images (1%)](./images/gan-1/samples_0002500.png)
![Synthesized images (1%)](./images/gan-1/samples_0005000.png)
![Synthesized images (1%)](./images/gan-1/samples_0010000.png)

1% data (2500, 5000, 10000 training episodes)

![MNIST classification error](./images/mnist-class-error.png)
