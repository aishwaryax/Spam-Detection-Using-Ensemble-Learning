from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from pylab import *
import itertools

def Ensemble_epoch(history_):
    Number_of_models = len(history_)
    caption=[]
    for i in range(0,len(history_)):
        caption.append('RDL '+str(i+1))
    plt.legend(caption, loc='upper right')
    for i in range(0, Number_of_models):
        plt.plot(history_[i].history['accuracy'])
        plt.title('model train accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    plt.legend(caption, loc='upper right')
    plt.show()



    for i in range(0, Number_of_models):
        plt.plot(history_[i].history['val_accuracy'])
        plt.title('model test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

    plt.show()
    plt.legend(caption, loc='upper right')
    for i in range(0, Number_of_models):
        # summarize history for loss
        plt.plot(history_[i].history['loss'])

        plt.title('model train loss ')
        plt.ylabel('loss')
        plt.xlabel('epoch')

    plt.legend(caption, loc='upper right')
    plt.show()
    plt.legend(
        caption, loc='upper right')
    for i in range(0, Number_of_models):
        # summarize history for loss
        plt.plot(history_[i].history['val_loss'])

        plt.title('model loss test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
    plt.legend(caption, loc='upper right')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def accuracy(y_test,final_y):
    np.set_printoptions(precision=2)
    y_test_temp = np.argmax(y_test, axis=1)
    F_score = accuracy_score(y_test_temp, final_y)
    F1 = precision_recall_fscore_support(y_test_temp, final_y, average='micro')
    F2 = precision_recall_fscore_support(y_test_temp, final_y, average='macro')
    F3 = precision_recall_fscore_support(y_test_temp, final_y, average='weighted')
    print(F_score)
    print(F1)
    print(F2)
    print(F3)
