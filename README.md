## RFI-Inpainting-with-DSS-Layers
Will update this read me in due time. Meanwhile, here is a quick description of the project.
### Tentative project description
One of the main challenges in measuring radio photons using ground instruments is the frequent data flagging
found due to radio frequency interference (RFI). PhD candidate Michael Pagano has been working on a
machine learning implementation of convolutions neural network (CNN) dictating the in-painting used to
circumvent the data-flagging caused by RFIs during the measurements of the HERA instrument. Given that
the HERA solely point upwards, there is some symmetry to be observed and leveraged in the data. Every
24 hours the instrument measures the same point in the sky. We can thus use the fact similar measurements
are made daily to better the in-painting behaviors. For example, if some flagging appeared in day 2 but
did not appeared in day 1, we should account for that within the network. Modifying the network used in
Pagano’s work using Deep Sets for Symmetric elements layers (DDS) could help improve the behaviors of
the machine learning implementation used to overcome RFIs found when measuring radio photons.
