

# extract add/delete lines from diff
import re
from unidiff import PatchSet


def extract_add_and_del_lines(diff, line_separator=' ', join=True):
    # line = line.strip()
    code_lines = diff.split('<nl>') # NOTE: stop removing file names // [2:]  # drop edit(add/delete) file names
    code_lines = [l.strip() for l in code_lines]
    if code_lines[-1] == '':  # drop last empty line
        code_lines = code_lines[:-1]

    add_lines = []
    del_lines = []
    for line in code_lines:
        if line.startswith('+') or line.startswith('mmm a'):
            if len(line) > 1:  # drop empty add line
                add_lines.append(line[1:])  # drop char: +
        elif line.startswith('-') or line.startswith('ppp b'):  # drop empty delete line
            if len(line) > 1:
                del_lines.append(line[1:])  # drop char: -
        else:  # add common line to both code line lists
            add_lines.append(line)
            del_lines.append(line)

    if join:
        return line_separator.join(add_lines), line_separator.join(del_lines)
    else:
        return add_lines, del_lines

# extract add/delete lines from diff
# V2:
# 1. drop mmm and ppp head lines
def extract_add_and_del_lines_v2(diff,
                                 line_separator=' ',
                                 join=True):
    # line = line.strip()
    code_lines = diff.split('<nl>') # NOTE: stop removing file names // [2:]  # drop edit(add/delete) file names
    code_lines = [l.strip() for l in code_lines]
    if code_lines[-1] == '':  # drop last empty line
        code_lines = code_lines[:-1]

    add_lines = []
    del_lines = []
    for line in code_lines:
        if line.startswith('mmm a') or line.startswith('ppp b'):
            continue
        elif line.startswith('+'):
            if len(line) > 1:  # drop empty add line
                add_lines.append(line[1:])
        elif line.startswith('-'):  # drop empty delete line
            if len(line) > 1:
                del_lines.append(line[1:])
        else:  # add common line to both code line lists
            add_lines.append(line)
            del_lines.append(line)

    if join:
        return line_separator.join(add_lines), line_separator.join(del_lines)
    else:
        return add_lines, del_lines

# extract add/delete lines from diff
# V3:
# 1. add mmm line to added lines, ppp line to del lines
def extract_add_and_del_lines_v3(diff,
                                 line_separator=' ',
                                 join=True):
    # line = line.strip()
    code_lines = diff.split('<nl>') # NOTE: stop removing file names // [2:]  # drop edit(add/delete) file names
    code_lines = [l.strip() for l in code_lines]
    if code_lines[-1] == '':  # drop last empty line
        code_lines = code_lines[:-1]

    add_lines = []
    del_lines = []
    for line in code_lines:
        if line.startswith('mmm a'):
            add_lines.append(line)
        elif line.startswith('ppp b'):
            del_lines.append(line)
        elif line.startswith('+'):
            if len(line) > 1:  # drop empty add line
                add_lines.append(line[1:])
        elif line.startswith('-'):  # drop empty delete line
            if len(line) > 1:
                del_lines.append(line[1:])
        else:  # add common line to both code line lists
            add_lines.append(line)
            del_lines.append(line)

    if join:
        return line_separator.join(add_lines), line_separator.join(del_lines)
    else:
        return add_lines, del_lines

from utils import GlobalLogger as mylogger

def extract_add_and_del_lines_v4(diff,
                                 line_separator=' ',
                                 join=True):
    """
    Include binary files, new/del files, new/old mode and similarity index.
    """
    all_lines = [l.strip() for l in diff.split('<nl>')]
    add_lines, del_lines = [], []

    i = 0
    while i < len(all_lines):
        line = all_lines[i]
        lowered_line = line.lower()
        if lowered_line.startswith('binary files'):
            file_line = line[12:]   # drop binary files string
            files = [l.strip() for l in file_line.split(' and ')]
            if len(files) != 2:
                mylogger.error('process_diff',
                               f"! Not two files in binary file line: {line}")
            else:
                add_lines.append('Binary files ' + files[0])
                del_lines.append('Binary files ' + files[1])
            i += 1
        elif lowered_line.startswith('similarity index'):
            if i + 2 >= len(all_lines):
                mylogger.error('process_diff',
                               f'! No more two lines after similarity index for diff: {diff}')
                break
            else:
                file_from, file_to = all_lines[i+1], all_lines[i+2]
                if not file_from.startswith('rename from') or not file_to.startswith('rename to'):
                    mylogger.warning('process_diff',
                                     f'% Similarity index is not rename operation: {file_from}, {file_to}')
                else:
                    del_lines.append(line + ' ' + file_from)
                    add_lines.append(line + ' ' + file_to)
                i += 3
        elif lowered_line.startswith('new file mode'):
            if i + 1 >= len(all_lines):
                mylogger.error('process_diff',
                               f'! No more one lines after new file mode for diff: {diff}')
                break
            else:
                if not all_lines[i+1].startswith('index '):
                    mylogger.warning('process_diff',
                                     f"% new file mode's next line not has index head: {all_lines[i+1]}")
                else:
                    add_lines += [line, all_lines[i], all_lines[i+1]]
                i += 2
        elif lowered_line.startswith('delete file mode'):
            if i + 1 >= len(all_lines):
                mylogger.error('process_diff',
                               f'! No more one lines after delete file mode for diff: {diff}')
                break
            else:
                if not all_lines[i+1].startswith('index '):
                    mylogger.warning('process_diff',
                                     f"% delete file mode's next line not has index head: {all_lines[i+1]}")
                else:
                    del_lines += [line, all_lines[i], all_lines[i+1]]
                i += 2
        elif lowered_line.startswith('old mode'):
            if i + 1 >= len(all_lines):
                mylogger.error('process_diff',
                               f'! No more one lines after old mode for diff: {diff}')
                break
            else:
                old_mode, new_mode = all_lines[i], all_lines[i+1]
                del_lines.append(old_mode)
                add_lines.append(new_mode)
                i += 2
        # Else code diff
        elif line.startswith('mmm ') or line.startswith('- '):
            del_lines.append(line[1:])
            i += 1
        elif line.startswith('ppp ') or line.startswith('+ '):
            add_lines.append(line[1:])
            i += 1
        else:
            del_lines.append(line)
            add_lines.append(line)
            i += 1

    add_lines = [l.strip() for l in add_lines]
    del_lines = [l.strip() for l in del_lines]

    if join:
        return line_separator.join(add_lines), line_separator.join(del_lines)
    else:
        return add_lines, del_lines


def extract_add_and_del_lines_fira_v1(diff,
                                      line_separator='\n',
                                      join=True):
    add_lines, del_lines = [], []
    patchset = PatchSet(diff)
    for file in patchset:
        for hunk in file:
            for line in hunk:
                line_val = line.value.strip()
                line_val = re.sub(r' +|\t+', ' ', line_val)
                # Filter empty line
                # if line_val == '':
                #     continue
                if line.is_added:
                    add_lines.append(line_val)
                elif line.is_removed:
                    del_lines.append(line_val)
                else:
                    add_lines.append(line_val)
                    del_lines.append(line_val)

    if join:
        return line_separator.join(add_lines), line_separator.join(del_lines)
    else:
        return add_lines, del_lines

def extract_add_and_del_lines_fira_v2(diff, **kwargs):
    if len(kwargs) > 0:
        mylogger.warning('extract_add_and_del_lines_fira_v2',
                         f'Useless kwargs given to fira v2 extractor: {kwargs}')
    diff_tokens = diff['tokens']
    diff_marks = diff['marks']
    add_tokens, del_tokens = [], []

    for token, mark in zip(diff_tokens, diff_marks):
        if mark == 1 or mark == 2:
            del_tokens.append(token)
        if mark == 3 or mark == 2:
            add_tokens.append(token)

    return ' '.join(add_tokens), ' '.join(del_tokens)

_version_mapping = {
    'v1': extract_add_and_del_lines,
    'v2': extract_add_and_del_lines_v2,
    'v3': extract_add_and_del_lines_v3,
    'v4': extract_add_and_del_lines_v4,
    'fira_v1': extract_add_and_del_lines_fira_v1,
    'fira_v2': extract_add_and_del_lines_fira_v2,
}

def get_line_extractor(version):
    return _version_mapping[version]