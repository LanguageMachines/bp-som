/* SOM.H */

float count_distance(int l, float min_dis, float *v)
{
   int i;
   float hulp=0.0;
   for (i=1; i<=n[l]; i++) {
     hulp += (som_vector[l][i] - v[i]) * (som_vector[l][i] - v[i]);
     if (hulp>min_dis) { break; }
   }
   return (float) sqrt((double) hulp);
}

void process_som_vectors(int lrn_yn, int material)
{
  int l, i, j, win_x, win_y, update_context;
  float distance, min_distance, min_part_win_distance;
  for (l=0;l<=LAST_LAYER;l++) {
    part_win_x[l]=part_win_y[l]=winner_labeln[l]=win_x=win_y=0;
    if (DIM_SOM[l]>0) {
      min_distance=min_part_win_distance=99999.0;
      for (i=1;i<=n[l];i++) { som_vector[l]=act[l]; }
      if (DEBUG) {
        printf("\nSOM-vector layer %d: ", l);
        for (i=1;i<=n[l];i++) {
          printf("%4.2f ", som_vector[l][i]);
        }
      }
      for (i=1;i<=DIM_SOM[l];i++) {
        for (j=1;j<=DIM_SOM[l];j++) {
	  /* result of distance is not always the real distance: the
	     calculation proces is aborted when
	     distance>min_part_win_distance */
	  distance=
	    count_distance(l, min_part_win_distance,
			   som_network_vector[l][i][j]);
          /* yes no min_distance */
          if (distance<min_distance) {
	    win_x=i; win_y=j;
            min_distance=distance;
	    winner_labeln[l]=(int) som_cell_teller[l][i][j][0];
          }
          /* is the winner the best partial som-winner? */
          if ((som_cell_teller[l][i][j][0]==labeln) &&
              (distance<=min_part_win_distance)) {
            part_win_x[l]=i; part_win_y[l]=j;
            min_part_win_distance=distance;
          }
        }
      }

      /* book keeping and results */
      som_total_distance[l]+=min_distance;
      winner_reliability[l]=som_winner_percents[l][win_x][win_y];
      winner_dis[l]=min_distance;
      if ((part_win_x[l]>0) &&
          (part_win_y[l]>0)) {
        part_winner_reliability[l] =
          som_winner_percents[l][part_win_x[l]][part_win_y[l]];
        use_som_error[l]=1;
      } else {
        part_winner_reliability[l]=0;
        use_som_error[l]=0;
      }
      part_winner_dis[l]=min_part_win_distance;

      if ((DEBUG==1) && (epoch>TEST_RATIO)) {
        printf("\nWinn  =[%d, %d], Min_D=%6.3f, Reliability=%d",
          win_x, win_y, winner_dis[l], winner_reliability[l]);
        printf("\nP_Winn=[%d, %d], Par_D=%6.3f, Reliability=%d",
          part_win_x[l], part_win_y[l],
          part_winner_dis[l],
          part_winner_reliability[l]);
        if (use_som_error[l]>0) { printf(" use SOM-error!"); }
      }

      if ((lrn_yn==1) && (win_x>0) &&
	  (win_y>0) && (min_distance>DONOTHING)) {
        for (i=win_x-som_cont; i<=win_x+som_cont; i++) {
          for (j=win_y-som_cont; j<=win_y+som_cont; j++) {
            if (i>=1 && j>=1 && i<=DIM_SOM[l] && j<=DIM_SOM[l]) {
              if (abs(i-win_x)>abs(j-win_y)) {
               update_context=abs(i-win_x);
              } else {
               update_context=abs(j-win_y);
              }
              update_som_network(l, i, j, update_context);
            }
          }
        }
      }

      if (material==1) {
	if ((ONLINE_CLASS_LABELING>0) &&
	    (epoch>TEST_RATIO)) {
 	  som_cell_teller[l][win_x][win_y][labeln]++;
	}
	if ((lrn_yn==0) && (ONLINE_CLASS_LABELING==0)) {
	  som_cell_teller[l][win_x][win_y][labeln]++;
	}
      } else {
        if (winner_labeln[l]==labeln) { som_ok[l]++; }
	som_conf_matrix[l][labeln][winner_labeln[l]]++;
      }
    }
  }
}

void show_som_labeling (void)
{
  int i, j, l;
  long int n_examples;
  char hulp[LABELL+1];
  float weighted_reliability;
  for (l=0;l<=LAST_LAYER;l++) {
    if (DIM_SOM[l]>0) {
      weighted_reliability=0.0; n_examples=0;
      printf("\n\nClass labeling SOM: %d\n\n   ", l);
      for (i=1; i<=DIM_SOM[l]; i++) {
        printf("%4d", i);
      }
      for (i=1; i<=DIM_SOM[l]; i++) {
        /* reliability-line */
        printf("\n   ");
        for (j=1; j<=DIM_SOM[l]; j++) {
          printf(" %3d", som_winner_percents[l][i][j]);
          weighted_reliability += 0.01 * som_winner_percents[l][i][j] *
	    som_cell_teller[l][i][j][(long int) som_cell_teller[l][i][j][0]];
	  n_examples += som_cell_teller[l][i][j][(long int) som_cell_teller[l][i][j][0]];

        }
        /* label-line */
        printf("\n%3d", i);
        for (j=1; j<=DIM_SOM[l]; j++) {
              if (som_cell_teller[l][i][j][0]>0) {
		strcpy(hulp, take_label((int) som_cell_teller[l][i][j][0]));
              } else {
                strcpy(hulp, "   ");
              }
              printf(" %3s", hulp);
        }
        /* class-count-line */
        printf("\n   ");
        for (j=1; j<=DIM_SOM[l]; j++) {
	  printf(" %3ld",
	   som_cell_teller[l][i][j][(long int) som_cell_teller[l][i][j][0]]);
        }
      }
      if (n_examples>0) {
        printf("\nweighted reliability: %6.2f \n",
	       (100.0*weighted_reliability)/(1.*n_examples));
      } else {
        printf("\nweighted reliability: n_examples=0 !!!\n");
      }
    }
  }
}

void record_som_labeling (void)
{
  int i, j, l;
  long int n_examples; float weighted_reliability;
  char hulp[LABELL+1];
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      weighted_reliability=0.0; n_examples=0;
      fprintf(fplog, "\n\nClass labeling SOM: %d\n\n   ", l);
      for (i=1; i<=DIM_SOM[l]; i++) {
        fprintf(fplog, "%4d", i);
      }
      for (i=1; i<=DIM_SOM[l]; i++) {
        /* label-line */
        fprintf(fplog, "\n   ");
        for (j=1; j<=DIM_SOM[l]; j++) {
          fprintf(fplog, " %3d", som_winner_percents[l][i][j]);
          weighted_reliability += 0.01 * som_winner_percents[l][i][j] *
	    som_cell_teller[l][i][j][(int) som_cell_teller[l][i][j][0]];
	  n_examples += som_cell_teller[l][i][j][(int) som_cell_teller[l][i][j][0]];

        }
        /* reliability-line */
        fprintf(fplog, "\n%3d", i);
        for (j=1; j<=DIM_SOM[l]; j++) {
          if (som_cell_teller[l][i][j][0]>0) {
	    strcpy(hulp, take_label((int) som_cell_teller[l][i][j][0]));
          } else {
            strcpy(hulp, "   ");
          }
          fprintf(fplog, " %3s", hulp);
        }
        /* label-count-line */
        fprintf(fplog, "\n   ");
        for (j=1; j<=DIM_SOM[l]; j++) {
	  fprintf(fplog, " %3ld",
		  som_cell_teller[l][i][j][(int) som_cell_teller[l][i][j][0]]);
        }
      }
      if (n_examples>0) {
        fprintf(fplog, "\nweighted reliability: %6.2f\n",
                100.0*weighted_reliability/n_examples);
      } else {
        fprintf(fplog, "\nweighted reliability: n_examples=0 !!!\n");
      }
    }
  }
}

void update_som_network(int l, int x, int y, int dist)
{
  int i;
  if (DEBUG) {
    printf("\nupdate %d %d %d distance=%d update factor=%5.2f",
           l, x, y, dist, 1.0/pow((double) 2, (double) dist));
  }
  for (i=1; i<=n[l]; i++) {
    som_network_vector[l][x][y][i] +=
      (float) (som_lr/pow((double) 2, (double) dist)) *
            (som_vector[l][i]-som_network_vector[l][x][y][i]);
  }
}

void clear_som_cell_counters(void)
{
  int ll, i, j, n;
  for (ll=0; ll<=LAST_LAYER; ll++) {
    for (i=1; i<=DIM_SOM[ll]; i++) {
      for (j=1; j<=DIM_SOM[ll]; j++) {
        som_winner_percents[ll][i][j]=0;
        for (n=0; n<=n_labels; n++) {
          som_cell_teller[ll][i][j][n]=0;
        }
      }
    }
  }
}

void show_som_conf_matrix(void)
{
  int i,j,l;
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      printf("\n\nSOM-classification(%d)-confusion matrix: %5.2f\n\n        ",
        l, (float) (100.0*som_ok[l]/example));

      for (i=1;i<=n_labels;i++) {
        printf("%s   ", take_label(i));
      }
      for (i=1;i<=n_labels;i++) {
        printf("\n%s  ", take_label(i));
        for (j=1;j<=n_labels;j++) {
              printf(" %5ld", som_conf_matrix[l][i][j]);
        }
      }
      printf("\n\n");
    }
  }
}

void record_som_conf_matrix(void)
{
  int i,j,l;
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      fprintf(fplog, "\n\nSOM-classification(%d)-confusion matrix: %5.2f\n\n        ",
        l, (float) (100.0*som_ok[l]/example));
      for (i=1;i<=n_labels;i++) {
        fprintf(fplog, "%s   ", take_label(i));
      }
      for (i=1;i<=n_labels;i++) {
        fprintf(fplog, "\n%s  ", take_label(i));
        for (j=1;j<=n_labels;j++) {
              fprintf(fplog, " %5ld", som_conf_matrix[l][i][j]);
        }
      }
      fprintf(fplog, "\n\n");
    }
  }
}

char *take_label(int n)
{
  int  i, pos;
  strcpy(helplabels, "???");
  if (n>=1) {
    pos=(n-1) * (LABELL+1);
    for (i=0; i<LABELL; i++) {
      helplabels[i]=ALL_LABELS[pos + i];
    }
  }
  return helplabels;
}

void count_som_cell_winners(void)
{
  int ll, i, j, l;
  float corrected_total, best_n, corrected_n;
  for (ll=0; ll<=LAST_LAYER; ll++) {
    if (DIM_SOM[ll]>0) {
      for (i=1; i<=DIM_SOM[ll]; i++) {
	for (j=1; j<=DIM_SOM[ll]; j++) {
	  best_n=corrected_total=0.0;
	  som_cell_teller[ll][i][j][0]=0;
	  for (l=1; l<=n_labels; l++) {
	    if (bias[l]>DONOTHING) {
	      corrected_n=(1.0*som_cell_teller[ll][i][j][l]/bias[l]);
	      corrected_total+=corrected_n;
	      if (corrected_n>best_n) {
		som_cell_teller[ll][i][j][0]=l;
		best_n=corrected_n;
	      }
	    }
	  }
	  if (corrected_total>0.0) {
	    som_winner_percents[ll][i][j]=
	      (int) (100.0 * pow((double) (best_n/corrected_total),
		      (double) BIAS_CORRECTION));

	  } else {
	    som_winner_percents[ll][i][j]=0;
	  }
	}
      }
      if (DEBUG) {
	for (i=1; i<=DIM_SOM[ll]; i++) {
	  for (j=1; j<=DIM_SOM[ll]; j++) {
	    printf("\n");
	    for (l=0; l<=n_labels; l++) {
	      printf("%3ld ", som_cell_teller[ll][i][j][l]);
	    }
	  }
	}
	printf("\n");
      }

      if (RECORD && DEBUG) {
	for (i=1; i<=DIM_SOM[ll]; i++) {
	  for (j=1; j<=DIM_SOM[ll]; j++) {
	    fprintf(fplog, "\n");
	    for (l=0; l<=n_labels; l++) {
	      fprintf(fplog, "%3ld ", som_cell_teller[ll][i][j][l]);
	    }
	  }
	}
        fprintf(fplog, "\n");
      }
    }
  }
}

void clear_som_conf_matrix(void)
{
  int l,i,j;
  for (l=0; l<=LAST_LAYER; l++) {
    for (i=1;i<=n_labels;i++) {
      for (j=1;j<=n_labels;j++) {
        som_conf_matrix[l][i][j]=0;
      }
    }
  }
}

void show_soms_epoch_score(void)
{
  int l;
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      printf(
        "\n%4d, layer %1d, ave_dist %6.3f, cont %d, l_r %6.3f, exa %5ld: ",
         epoch, l, som_total_distance[l]/example, som_cont, som_lr, example);
    }
  }
}

void record_soms_epoch_score(void)
{
  int l;
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      fprintf(fplog,
        "\n%4d, layer %1d, ave_dist %6.3f, cont %d, l_r %6.3f, exa %5ld: ",
         epoch, l, som_total_distance[l]/example, som_cont, som_lr, example);
    }
  }
}

void div_class_count()
{
  int l, x, y, n;
  for (l=0; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      for (x=1; x<=DIM_SOM[l]; x++) {
        for (y=1; y<=DIM_SOM[l]; y++) {
          for (n=1; n<=n_labels; n++) {
            som_cell_teller[l][x][y][n]/=CLASS_COUNT_DIV;
          }
        }
      }
      /* count_som_cell_winners(l); */
    }
  }
}
