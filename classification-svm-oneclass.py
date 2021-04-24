#membuat fungsi untuk menjalankan model svm oneclass
def svmoneclass(train,test,testlabel):
  from sklearn.svm import OneClassSVM
  model = OneClassSVM(kernel='rbf',gamma="scale", nu=0.01)
  model.fit(train)
  y_pred_train = model.predict(train)
  y_pred_test = model.predict(test)
  #y_pred_outliers = clf.predict(X_outliers)
  n_error_train = y_pred_train[y_pred_train == -1].size
  #n_error_test = y_pred_test[y_pred_test == -1].size
  #n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size'''
  from sklearn import metrics
  preds =  model.predict(test)
  targs = testlabel
  print("accuracy: ", metrics.accuracy_score(targs,preds))
  print("precision: ", metrics.precision_score(targs, preds)) 
  print("recall: ", metrics.recall_score(targs,preds))
  print("f1: ", metrics.f1_score(targs, preds))
  print("area under curve (auc): ", metrics.roc_auc_score(targs,preds))

#mejalankan model pada fitur  
svmoneclass(TrainmfccJangkrik,test_mfcc,TestLabel_mfcc)
svmoneclass(TrainlfccJangkrik,test_lfcc,TestLabel_lfcc)
svmoneclass(TrainmfcclJangkrik,test_mfccl,TestLabel_mfccl)
