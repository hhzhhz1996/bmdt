import pefile
import os
from capstone import *

api_mapping = {}
with open('resources/api.txt') as f:
    for idx, api_name in enumerate(f.readlines()):
        api_mapping[api_name.strip()] = idx + 1
        api_mapping[str(idx + 1)] = api_name.strip()


def get_api_set_single(file_path, to_id=True):
    try:
        pe = pefile.PE(file_path)
    except Exception as e:
        e = TypeError("unsupported type or damaged file")
        e.s
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
            except Exception as e:
                continue
    return api_set


def get_api_set_batch(directory_path, to_id=True):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_sets = []
    for index, file in enumerate(file_ls):
        file_path = os.path.join(directory_path, file)
        try:
            api_ls = get_api_set_single(file_path)
            if len(api_ls) > 0:
                api_sets.append(api_ls)
                print(index)
        except Exception as e:
            print(file)

    return api_sets


def get_api_seq_single(file_path, arch=CS_ARCH_X86, mode=CS_MODE_32):
    pe = pefile.PE(file_path)
    eop = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    code_section = pe.get_section_by_rva(eop)
    code_dump = code_section.get_data()
    code_addr = pe.OPTIONAL_HEADER.ImageBase + code_section.VirtualAddress
    # api mapping
    api_mapping = {}
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                api_mapping[hex(imp.address)] = str(imp.name, encoding='utf-8')
                api_mapping[str(imp.name, encoding='utf-8')] = hex(imp.address)
            except Exception as e:
                pass

    md = Cs(arch, mode)
    md.detail = True
    md.skipdata = True
    md.skipdata_setup = ("db", None, None)

    api_call_seq = []

    for i in md.disasm(code_dump, code_section.VirtualAddress):
        # print("0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

        if i.mnemonic in ('call', 'jmp'):

            split = i.op_str.split(' ')
            if len(split) == 1:
                address = split[0]
            elif len(split) == 3:
                address = split[-1][1:-1]
            else:
                address = ''
            if address in api_mapping:
                if api_mapping[address] in api_name_to_id:
                    api_call_seq.append(api_name_to_id[api_mapping[address]])

    if len(api_call_seq) == 0:
        api_call_seq.append('0')

    return api_call_seq


def get_api_seq_batch(directory_path):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_call_seqs = []
    for index, file in enumerate(file_ls):
        try:
            api_call_seq = get_api_seq_single(os.path.join(directory_path, file), CS_ARCH_X86, CS_MODE_32)
            if len(api_call_seq) > 0:
                api_call_seqs.append(api_call_seq)
                print(index)
        except Exception as e:
            print(file)

    return api_call_seqs


def write_feature(lists, directory_path, file_name, append=False):
    action = 'a' if append else 'w'
    with open(os.path.join(directory_path, file_name), action) as wf:
        for line in lists:
            wf.write(' '.join(line))
            wf.write('\n')


if __name__ == '__main__':
    path = r'D:\actProject\files\unpack\malicious\test'
    output_path = r'C:\Users\shini\Desktop\api_sequence\unpack\test\malicious.txt'

    get_api_set_batch(directory_path=path)

