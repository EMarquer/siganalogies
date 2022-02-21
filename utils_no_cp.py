import random
"""File containing generic data manipulation methods for analogy-oriented datasets, when central permutation is not accepted."""

def enrich(a, b, c, d):
    """Apply the example generation process from 'Solving Word Analogies: a Machine Learning Perspective', modified to exclude central permutation."""
    yield a, b, c, d
    yield c, d, a, b
    yield d, c, b, a
    yield b, a, d, c

def generate_negative(a, b, c, d):
    """Apply the negative example generation process from 'Solving Word Analogies: a Machine Learning Perspective', modified to include central permutation."""
    #previously accepted by central perm
    yield a, c, b, d # implied by the enrichment d, b, c, a | b, d, a, c | c, a, d, b
    yield b, a, c, d
    yield c, b, a, d
    yield a, a, c, d

def random_sample_negative(a, b, c, d, filter_invalid=True, tensors=True, n=8):
    """Randomly sample invalid analogies from the available negative forms.
    
    In this variant, central permutation is not accepted.
    
    :param filter_invalid: If True, will exclude quadruples from negative forms if the an identical quadruple is already present in the positive forms.
    :param tensors: If True, consider that a, b, c, and d are tensors and use an appropriate equality check for quadruples.
    :param n: Number of valid and invalid forms to return. If None, all the forms will be returned, no matter the number.
    """
    # random sampling
    return n_pos_n_neg(a, b, c, d, filter_invalid=filter_invalid, tensors=tensors, n=n)[1]

def n_pos_n_neg(a, b, c, d, filter_invalid=True, filter_valid=False, tensors=True, n=8):
    """Generate positive examples (including the base form) and negative examples form a base analogical form.

    In this variant, central permutation is not accepted.

    Returns two lists, one of positive forms, the other of negative forms.

    :param filter_invalid: If True, will exclude quadruples from negative forms if the an identical quadruple is already present in the positive forms.
    :param filter_valid: If True, will exclude quadruples from positive forms if the an identical quadruple is already present in the negative forms.
        Cannot be True if filter_invalid is True
    :param tensors: If True, consider that a, b, c, and d are tensors and use an appropriate equality check for quadruples.
    :param n: Number of valid and invalid forms to return. If None, all the forms will be returned, no matter the number.
    """

    assert not (filter_invalid and filter_valid), "filter_invalid and filter_valid must not be True together."

    def tensor_in(a, b, c, d, list_of_form):
        # check if the quadruple is equal to a quadruple in the givenlist of forms
        return any(
            a.equal(a_) and
            b.equal(b_) and
            c.equal(c_) and
            d.equal(d_)
            for a_, b_, c_, d_ in list_of_form
        )

    # Case 1: filtering out invalid forms which are already present as valid forms
    if filter_invalid:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        
        negative_forms=[(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
            if (
                # remove potentially problematic forms (which can appear when the input is a:b:a:b for example)
                (tensors and not tensor_in(a__, b__, c__, d__, positive_forms)) or 
                (not tensors and (a__, b__, c__, d__) not in positive_forms)
            )
        ]

    # Case 2: filtering out valid forms which are already present as invalid forms
    elif filter_valid:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        
        negative_forms = [(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
        ]
        positive_forms = [
            (a_, b_, c_, d_) for a_, b_, c_, d_ in positive_forms
            if (
                # remove potentially problematic forms (which can appear when the input is a:b:a:b for example)
                (tensors and not tensor_in(a_, b_, c_, d_, negative_forms)) or 
                (not tensors and (a_, b_, c_, d_) not in negative_forms)
            )
        ]

    # Case 3: no filtering
    else:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        negative_forms = [(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
        ]

    # Sampling the right number
    if n is None or len(positive_forms) == n:
        positive_forms = positive_forms
    elif len(positive_forms) > n:
        positive_forms = random.sample(positive_forms, n)
    else: # len(positive_forms) < n:
        positive_forms = positive_forms + random.choices(positive_forms, n - len(positive_forms))

    # Sampling the right number
    if n is None or len(negative_forms) == n:
        negative_forms = negative_forms
    elif len(negative_forms) > n:
        negative_forms = random.sample(negative_forms, n)
    else: # len(negative_forms) < n:
        negative_forms = negative_forms + random.choices(negative_forms, n - len(negative_forms))

    return positive_forms, negative_forms