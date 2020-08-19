API Classes
=======

.. toctree::
    :maxdepth: 2
    :caption: Contents:

I generally believe in functional programming paradigm, where methods or functions only depend on input itself and the data is immutable (meaning it would not change in the middle of the run). This greatly increases the modularity of the code. But I also like the inheritance of the python classes. So I can easily inherit one class from another and by doing that all the methods and functions are available to me for a complete different stuff. Putting all the necessary functions in a same class makes me easier to group them. Thus you will see the codes are neither completely functional or object oriented. It is in the middle of the both. For example, you will see all the classmethods here are simply functions. They do not share anything common or mutable object (class variables) and thus given the same input, will always produce the same output (regardless of the class). All the classes has a wrapper function, which is the main function of the class. It will hold all the variables inside it. Given the required input, it will go through all the necessary functions (classmethods) and will finally output the results. The wrapper function is always put inside the __new__ class method, thus every time you use the class, it will run the wrapper method of that class, meaning the class can be used like a function.

Classification
--------------

ABC_TFK_Classification
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Classification
    :members:
    :inherited-members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Classification_PreTrain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Classification_PreTrain
    :members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Classification_Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Classification_Train
    :members:
    :show-inheritance:
    :special-members: __new__


ABC_TFK_Classification_CV
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Classification_CV
    :members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Classification_After_Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Classes.ABC.ABC_TFK_Classification_After_Train
    :members:
    :show-inheritance:
    :special-members: __new__

Parameter Estimation
--------------------

ABC_TFK_Params
^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Params
    :members:
    :inherited-members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Params_PreTrain
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Params_PreTrain
    :members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Params_Train
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Params_Train
    :members:
    :show-inheritance:
    :special-members: __new__


ABC_TFK_Params_CV
^^^^^^^^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_Params_CV
    :members:
    :show-inheritance:
    :special-members: __new__

ABC_TFK_Params_After_Train
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Classes.ABC.ABC_TFK_Params_After_Train
    :members:
    :show-inheritance:
    :special-members: __new__

Parmeter Estimation by SMC
--------------------------

ABC_TFK_NS
^^^^^^^^^^

.. autoclass:: Classes.ABC.ABC_TFK_NS
    :members:
    :inherited-members:
    :show-inheritance:
    :special-members: __new__
