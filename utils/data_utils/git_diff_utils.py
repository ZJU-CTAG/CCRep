import re
from unidiff import PatchSet


def clean_diff_meta_data_v3(raw_diff: str, line_separator: str = ' '):
    def _get_file_name(_file_name):
        _file_name_split = _file_name.split('/')
        return '/'.join(_file_name_split[1:])

    cleaned_diff = ''
    patchset = PatchSet(raw_diff)
    for file in patchset:
        source_file_name = _get_file_name(file.source_file)
        target_file_name = _get_file_name(file.target_file)
        # If same last file name, only keep one file name in the diff
        if source_file_name == target_file_name:
            file_names = f'{source_file_name}{line_separator}'
        else:
            file_names = f'a/{source_file_name} b/{target_file_name}{line_separator}'
        cleaned_diff += file_names
        for hunk in file:
            for line in hunk:
                line_val = line.value.strip()
                line_val = re.sub(r' +|\t+', ' ', line_val)
                if line.is_added:
                    cleaned_diff += f'+ {line_val}{line_separator}'
                elif line.is_removed:
                    cleaned_diff += f'- {line_val}{line_separator}'
                else:
                    cleaned_diff += f'{line_val}{line_separator}'
    return cleaned_diff

def _not_clean(raw_diff: str, line_separator: str = ' '):
    return raw_diff

clean_diff_meta_data_map = {
    'v3': clean_diff_meta_data_v3,

    None: _not_clean,
    'none': _not_clean,
}

def get_clean_diff_meta_data_func(version):
    return clean_diff_meta_data_map.get(version, _not_clean)   # none in default, even missing key