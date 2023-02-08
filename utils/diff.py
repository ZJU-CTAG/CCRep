from tqdm import tqdm
from difflib import Differ


def make_diff_format_file_from_json_objs(objs, dump_file_path,
                                         diff_key, obj_attr_keys=[]):
    f = open(dump_file_path, 'w')
    obj_separetor = '='*100
    hunk_separetor = '*' + '-'*50

    for obj in tqdm(objs):
        for key in obj_attr_keys:
            f.write(f'{key}: {obj.get(key)}\n')

        for i, hunk in enumerate(obj[diff_key]):
            f.write(f'\n{hunk_separetor}\n\n')
            for line in hunk:
                if line.startswith('+ ') or line.startswith('- '):
                    f.write(f'{line}\n')
                elif not line.startswith('? '):
                    f.write(f'{line}\n')


        f.write(f'\n{obj_separetor}\n\n')

    f.close()


def remake_diff_from_hunks(hunks, line_joiner=' '):
    add_lines, del_lines = [], []
    for hunk in hunks:
        for line in hunk['added_code']:
            add_lines.append(line)
        for line in hunk['removed_code']:
            del_lines.append(line)
    differ = Differ()
    diffs = differ.compare(del_lines, add_lines)

    diff_lines = []
    for line in diffs:
        if not line.startswith('? '):
            diff_lines.append(line)
    return line_joiner.join(diff_lines)