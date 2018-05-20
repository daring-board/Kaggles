import sys, os

if __name__=="__main__":

    # f.write('%s: %d, %d, %d, '%(f_name, label, np.argmax(pred_class1[idx]), np.argmax(pred_class2[idx])))
    # f.write('%d, %d, %d\n'%(np.argmax(ems), np.argmax(tta_ems), np.argmax(all_ems)))

    header = ['ResNet', 'VGG16', 'model_ems', 'tta_ems', 'model_tta_ems']
    data = {l.strip().split(':')[0]: l.strip().split(':')[1].split(',') for l in open('validate.csv', 'r')}
    for idx in range(1, 6):
        n_correct = len([1 for key in data.keys() if data[key][0] == data[key][idx]])
        print('%s: %d / %d'%(header[idx-1], n_correct, len(data)))
