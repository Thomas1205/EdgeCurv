include common/Makefile.common
INCLUDE += -I common/ -I optimization/ -I thirdparty/

all : $(DEBUGDIR) $(OPTDIR) .subdirs freemotionts.opt.L64

freemotionts.opt.L64: $(OPTDIR)/ratio_segmentation.o $(OPTDIR)/curvature.o $(OPTDIR)/mesh2D.o $(OPTDIR)/gradient.o $(OPTDIR)/grid.o $(OPTDIR)/svg.o $(OPTDIR)/conversion.o $(OPTDIR)/draw_segmentation.o common/$(OPTDIR)/application.o common/$(OPTDIR)/vector.o common/$(OPTDIR)/matrix.o common/$(OPTDIR)/tensor.o common/$(OPTDIR)/makros.o common/$(OPTDIR)/stringprocessing.o common/$(OPTDIR)/fileio.o
	$(LINKER) $(OPTFLAGS) $(INCLUDE) mrseg.cc $(OPTDIR)/ratio_segmentation.o $(OPTDIR)/curvature.o $(OPTDIR)/mesh2D.o $(OPTDIR)/gradient.o $(OPTDIR)/grid.o $(OPTDIR)/svg.o $(OPTDIR)/conversion.o $(OPTDIR)/draw_segmentation.o common/$(OPTDIR)/application.o common/$(OPTDIR)/vector.o common/$(OPTDIR)/matrix.o common/$(OPTDIR)/tensor.o common/$(OPTDIR)/makros.o common/$(OPTDIR)/stringprocessing.o common/$(OPTDIR)/fileio.o -o $@

.subdirs :
	cd common; make; cd -; cd -

include common/Makefile.finish
    
clean:
	cd common; make clean; cd -
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o 

  