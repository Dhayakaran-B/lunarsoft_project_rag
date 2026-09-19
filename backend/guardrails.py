def has_sufficient_evidence(results):

    if not results:
        return False

    return True


def clean_answer(answer):

    if not answer:
        return "I don't know based on the provided document."

    return answer.strip()