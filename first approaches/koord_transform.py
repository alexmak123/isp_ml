#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().run_line_magic('cd', '../test_cappa_koef')


# In[19]:


get_ipython().run_line_magic('ls', '')


# In[20]:


get_ipython().run_line_magic('cd', 'anna_t_histology')


# In[21]:


get_ipython().run_line_magic('cd', 'labels/')


# In[22]:


get_ipython().run_line_magic('cd', 'anna_t_save')


# In[23]:


get_ipython().run_line_magic('ls', '')


# In[24]:


import os
PATH = os.getcwd()
print (PATH)


# In[1]:


res = []
for filename in os.listdir(PATH):
    curr_lines_for_file = []
    filepath = os.path.join(PATH, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            contents = file.read()
            # Split the text into lines using the splitlines() method
            lines = contents.splitlines()
            for i in range(len(lines)):
                words = lines[i].split()
                words[0] = str(round(float(words[0]) - 128, 1))
                words[1] = str(round(float(words[1]) - 128, 1))
                lines[i] = (words[0], words[1], words[2])
            curr_lines_for_file.append(lines)
            print (lines)
        # Close the file for reading
        file.close()
    res.append(curr_lines_for_file)
        


# In[26]:


i = 0
for filename in os.listdir(PATH):
    filepath = os.path.join(PATH, filename)
    print (filename, i)
    if os.path.isfile(filepath):
        with open(filepath, 'w') as file:
            print (i)
            for line in res[i][0]:
                curr_line = str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n"
                #print (curr_line)
                file.write(curr_line)
        file.close()
    i += 1


# In[ ]:




