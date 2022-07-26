
def load_lsf(file_name):
    """
    This function is used to load the provided script as string and stripe out all comments (#), notice the comment can only be written behind the code
    ---INPUTS---
    FILE_NAME:  string, relative address or absoulte address of the file
    """
    # collect the code lines
    with open(file_name, 'r') as f:
        script_list = []
        for line in f.read().splitlines():
            if not line.strip().startswith('#'):
                script_list.append(line.strip())

    # integrte the lines together
    script = ' '.join(script_list)
    if not script:
        raise UserWarning('empty script')
    return script
    
