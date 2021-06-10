This repo provide two preliminary methods to combine CNN and MLP to understand video.

• Resnet + single MLP **(resSLP)**

• Resnet + MLP based Attention **(ResMA)**

---

- run script

  ```python
  python run.py -m ModalName -b BatchSize -gd ClipGrad -g GpuId -lr 0.01 -d DataSet
  ```

- example

  ```python
  python run.py -m resLP -b 16 -g 1 -gd 40 -l 4 -lr 0.01 -d Kinetic
  ```

  

