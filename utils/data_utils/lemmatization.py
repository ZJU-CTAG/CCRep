
fira_lemmatization = {"added": "add",
                      "adding": "add",
                      "adds": "add",
                      "fixed": "fix",
                      "fixing": "fix",
                      "removed": "remove",
                      "removing": "remove"}

def apply_fira_lemmatization(raw_msg):
    tokens = raw_msg.split()
    new_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in fira_lemmatization:
            new_tokens.append(fira_lemmatization[token_lower])
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)