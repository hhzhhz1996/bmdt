import pefile
import os
import capstone
import math
import shutil
from constant import BYTE_STREAM_LENGTH, api_mapping


def get_api_set_single(file_path, to_id=True):
    try:
        pe = pefile.PE(file_path)
    except Exception:
        raise TypeError("unsupported type or damaged file")

    if pe.DIRECTORY_ENTRY_IMPORT is None:
        raise ValueError("no imports")

    api_set = []
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                name = str(imp.name, 'utf-8')
                if name in api_mapping:
                    if to_id:
                        api_set.append(api_mapping[name])
                    else:
                        api_set.append(name)
            except Exception:
                continue
    return api_set


def get_api_set_batch(directory_path, to_id=True):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_sets = []
    for index, file in enumerate(file_ls):
        file_path = os.path.join(directory_path, file)
        try:
            api_ls = get_api_set_single(file_path, to_id)
            api_sets.append(api_ls)
        except TypeError:
            continue
        except ValueError:
            api_sets.append([0])
    return api_sets


def get_api_seq_single(file_path, arch=capstone.CS_ARCH_X86, to_id=True):
    pe = pefile.PE(file_path)
    api_address = {}
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                addr, name = hex(imp.address), str(imp.name, encoding='utf-8')
                api_address[addr], api_address[name] = name, addr
            except Exception as e:
                pass
    if pe.FILE_HEADER.Machine == 332:
        mode = capstone.CS_MODE_32
    else:
        mode = capstone.CS_MODE_64
    md = capstone.Cs(arch, mode)
    md.detail = True
    md.skipdata = True
    md.skipdata_setup = ("db", None, None)

    api_call_seq = []

    for section in pe.sections:
        for i in md.disasm(section.get_data(), section.VirtualAddress):
            # print("0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))
            if i.mnemonic in ('call', 'jmp'):
                split = i.op_str.split(' ')
                if len(split) == 1:
                    address = split[0]
                elif len(split) == 3:
                    address = split[-1][1:-1]
                else:
                    address = ''
                if address in api_address:
                    name = api_address[address]
                    if name in api_mapping:
                        api_call_seq.append(api_mapping[name] if to_id else name)

    return api_call_seq


def get_api_seq_batch(directory_path):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_call_seqs = []
    for index, file in enumerate(file_ls):
        try:
            api_call_seq = get_api_seq_single(os.path.join(directory_path, file), to_id=True)
            if len(api_call_seq) > 0:
                api_call_seqs.append(api_call_seq)
            else:
                print(file)
        except Exception:
            print(file)

    return api_call_seqs


def get_byte_stream_single(file_path):
    with open(file_path, 'rb') as f:
        s = f.read(BYTE_STREAM_LENGTH)
        ls = []
        for _ in s:
            ls.append(_ + 1)  # 0作为填充  所以+1

        if len(ls) < BYTE_STREAM_LENGTH:
            ls += [0 for _ in range(BYTE_STREAM_LENGTH - len(ls))]
        return ls


def get_byte_stream_batch(directory_path):
    byte_streams = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        byte_stream = get_byte_stream_single(file_path)
        byte_streams.append(byte_stream)

    return byte_streams


def get_pack_check_features(file_path):
    pe = pefile.PE(file_path)
    key_api = ['LoadLibrary', 'GetProcAddress', 'LoadLibraryA', 'LoadLibraryW', 'LoadLibraryEx']

    def get_imports_num():
        cnt = 0
        try:
            for dll in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in dll.imports:
                    cnt += 1
        except:
            return 0
        return cnt

    def has_key_api():
        try:
            for dll in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in dll.imports:
                    if str(imp.name, 'utf-8') in key_api:
                        return 1
        except:
            return 0
        return 0

    def get_section_entropy():
        entropy_sections = []
        for section in pe.sections:
            entropy_sections.append(section.get_entropy())

        return max(entropy_sections), sum(entropy_sections) / len(entropy_sections)

    def get_section_size():
        ratios = []
        for section in pe.sections:
            if section.Misc_VirtualSize == 0:
                continue
            ratios.append(section.SizeOfRawData / section.Misc_VirtualSize)

        return min(ratios), sum(ratios) / len(ratios)

    def get_entropy():
        frequency = [0 for _ in range(256)]
        entropy = 0
        data = open(file_path, 'rb').read()
        for byte in data:
            frequency[byte] += 1

        for k in range(256):
            p = frequency[k] / len(data)
            if p == 0.0:
                entropy += 0
            else:
                entropy += -1 * p * math.log2(p)

        return frequency[0] / len(data), entropy

    min_ratio, avg_ratio = get_section_size()
    max_entropy, avg_entropy = get_section_entropy()
    zero_frequency, entire_entropy = get_entropy()

    feature = [get_imports_num(), has_key_api(), avg_ratio, min_ratio,
               max_entropy, avg_entropy, entire_entropy, zero_frequency]
    return feature


def dump_feature(lists, directory_path, file_name, append=False):
    action = 'a' if append else 'w'
    with open(os.path.join(directory_path, file_name), action) as wf:
        for line in lists:
            wf.write(','.join(line))
            wf.write('\n')


if __name__ == '__main__':

    demo_file_path = r'D:\actProject\files\unpack\malicious\test' \
                r'\0421df2f603ce755ccef222281eca5de0cf804e42576204033d30dbf1006049e '
    path = r'C:/users/shini/desktop/benign'

    get_api_seq_batch(path)
