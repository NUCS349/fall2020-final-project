def test_imports():
    """
    Please don't import sklearn.feature_extraction to solve any of the problems in this assignment. 
    If you fail this test, we will give you a zero for this assignment.

    the 'a' in the file name is so this test is run first on a clean Python interpreter.
    """
    import sys
    import src
    assert 'sklearn.feature_extraction' not in sys.modules.keys()
