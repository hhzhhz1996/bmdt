import pefile
import os
import capstone

api_mapping = {}
with open('../resources/api.txt') as f:
    for idx, api_name in enumerate(f.readlines()):
        api_name = api_name.strip()
        api_mapping[api_name] = idx + 1  # index 0 reserves for padding
        api_mapping[str(idx + 1)] = api_name


def get_api_set_single(file_path, to_id: 'list api with name or id' = True):
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


def get_api_seq_single(file_path, arch=capstone.CS_ARCH_X86, mode=capstone.CS_MODE_32, to_id=True):
    pe = pefile.PE(file_path)
    eop = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    code_section = pe.get_section_by_rva(eop)
    code_dump = code_section.get_data()
    code_addr = pe.OPTIONAL_HEADER.ImageBase + code_section.VirtualAddress
    api_address = {}
    cnt = 0
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                addr, name = hex(imp.address), str(imp.name, encoding='utf-8')
                api_address[addr], api_address[name] = name, addr
                cnt += 1
            except Exception as e:
                pass
    print(cnt)

    md = capstone.Cs(arch, mode)
    md.detail = True
    md.skipdata = True
    md.skipdata_setup = ("db", None, None)

    api_call_seq = []

    for i in md.disasm(code_dump, code_section.VirtualAddress):
        # print("0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

        if i.mnemonic == 'call':
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
            api_call_seq = get_api_seq_single(os.path.join(directory_path, file))
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
    # for test
    demo_file_path = r'D:\actProject\files\unpack\malicious\test' \
                r'\4810283ba180366087af09329069f8767ba2a4e57fadfb31bab8827d4f90c6cd '
    path = r'D:\actProject\files\unpack\malicious\test'

    # get_disassemble(demo_file_path)
    single = get_api_seq_single(demo_file_path, to_id=False)
    print('\n'.join(single))
